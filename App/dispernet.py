import requests
import numpy as np
import sklearn.cluster as sc
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
import h5py
import os
import re
import scipy.ndimage as sn

np.set_printoptions(suppress=True)

def pick(spec, threshold=0.5, freq=[0.,0.3], velo=[2000, 6000], net='noise', errorbar=False, flipUp=False, searchStep=10, searchBorder=0, returnSpec=False, ind=-1, url = 'http://10.20.43.106:8514'):
	
	fMax = max(freq)
	fMin = min(freq)
	cMax = max(velo)
	cMin = min(velo)
	
	if fMax == fMin:
		raise ValueError('freq must be a range or array')
	
	if cMax == cMin:
		raise ValueError('velo must be a range or array')
		
	if threshold >= 1 or threshold <= 0:
		raise ValueError('threshold must in range (0,1)')
		
	if searchBorder >= 0.5 or searchBorder < 0:
		raise ValueError('searchBorder must in range [0,0.5)')
	
	if spec.ndim != 2:
		raise ValueError('the input spectrum must be a 2-D matrix')
	
	fileName = "spec.csv"
	if ind != -1:
		fileName = str(id) + fileName
	
	np.savetxt(fileName, np.squeeze(spec), delimiter=',')
	files = {'file': open(fileName, 'rb')}
	
	if errorbar == False:
		errorbar = '0'
	else:
		errorbar = '1'
		
	if flipUp == False:
		flipUp = '0'
	else:
		flipUp = '1'
		
	if returnSpec == True:
		returnSpec = 'spec'
	else:
		returnSpec = 'curve'
	
	info = {
			'fMin': str(fMin),
			'fMax': str(fMax),
			'cMin': str(cMin),
			'cMax': str(cMax),
			'threshold':str(threshold),
			'errorBar':errorbar,
			'dataType':net,
			'flipUp':flipUp,
			'searchStep':str(searchStep),
			'searchBorder':str(searchBorder),
			'returnType':returnSpec,
		}
	response = requests.post(url, data=info, files=files)
	
	if returnSpec == 'spec':
		output = np.array([float(x) for x in response.text.split()]).reshape((512,512))
		output = np.flip(output,0)
	else:
		if errorbar == 'curve':
			output = np.array([float(x) for x in response.text.split()]).reshape((-1,4))[:, [1,0,2,3]]
		else:
			output = np.array([float(x) for x in response.text.split()]).reshape((-1,2))[:, [1,0]]
			output=output
		
	os.remove(fileName)
	return output
	
def modeSeparation(curves, modes=2):
	curve_whiten = whiten(curves[:,0:2])      
	cluster_pred = sc.AgglomerativeClustering(n_clusters=int(modes),linkage='single',compute_full_tree=True).fit_predict(curve_whiten)
	
	m_value = np.zeros(modes)
	
	for mode in range(modes):
		m_value[mode] = np.mean(curves[cluster_pred==mode])
	
	m_c = np.vstack([m_value, np.arange(modes)])

	m_c = m_c[:, m_c[0,:].argsort()]

	cluster_out = cluster_pred.copy()
	for mode in range(modes):

		cluster_out[cluster_pred==m_c[1,mode]] = mode
	
	if curves.shape[1] == 2 or curves.shape[1] == 4:
		out = np.column_stack([curves, cluster_out])
	
	else:
		out = np.column_stack([curves[:, 0:-1], cluster_out])
	
	out = out[np.argsort(out[:,-1])]
	for mode in range(modes):
		curveInMode = out[out[:,-1] == modes]
		out[out[:,-1] == modes] = curveInMode[np.argsort(curveInMode[:,0])]
		
	return out
	
def autoSeparation(curves):
	
	fMax = max(curves[:,0])
	fMin = min(curves[:,0])
	cMax = max(curves[:,1])
	cMin = min(curves[:,1])
	
	fSearchStart = 0.05 * (fMax - fMin) + fMin
	cJumpRangeLimit = 0.1 * (cMax - cMin)
	
	exitFlag = False
	for modePre in range(5):
		curvePre = modeSeparation(curves, int(modePre)+1)
		for mode in range(modePre+1):
			curveInMode = curvePre[curvePre[:,-1] == mode]
			curveInMode = curveInMode[curveInMode[:,0]> fSearchStart]
			curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
			
			if np.std(np.diff(curveInMode[:,1])) > cJumpRangeLimit:
				exitFlag = False
				break
			else:
				exitFlag = True
				
		if exitFlag:
			break
	
	return curvePre
	
def discPointRemove(curve, threshold=2):
	curveRemoved = []
	for mode in range(int(max(curve[:,-1]))+1):
		curveInMode = curve[curve[:,-1]==mode]
		if len(curveInMode) > threshold:
			curveRemoved.append(curveInMode)
			
	curveRemoved = np.vstack(curveRemoved)
	return curveRemoved
		
	
def show(spec,curve,freq=[0.,0.3], velo=[2000, 6000], unit='m/s', s=10, ax=[], holdon=False, cmap='viridis', vmin=None, vmax=None):
	fMax = max(freq)
	fMin = min(freq)
	cMax = max(velo)
	cMin = min(velo)
	
	if ax == []:
		ax = plt.gca()
			
	if fMax == fMin:
		raise ValueError('freq must be a range or array')
	
	if cMax == cMin:
		raise ValueError('velo must be a range or array')
		
	ax.imshow(np.flip(spec,0),aspect='auto', extent=[fMin, fMax, cMin, cMax], cmap=cmap, vmin=vmin, vmax=vmax)
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Phase Velocity ('+unit+')')
	
	markerList=['*', 'o','v','^','s']
	
	if len(curve)>0:
		if curve.shape[1] == 2 or curve.shape[1] == 4:
			ax.scatter(curve[...,0],curve[...,1],s=s, edgecolors='w')
		else:
			for ii in range(int(max(curve[...,-1]))+1):
				curve_in_mode = curve[curve[:,-1] == ii]
				ax.scatter(curve_in_mode[...,0], curve_in_mode[...,1],label='mode '+str(ii),s=s,edgecolors='k',marker=markerList[ii])
			
			ax.legend()
	
	if not holdon:
		plt.show()
		
		
def save2h5(spectrum, freq, velo, fileName=''):
	if fileName == '':
		fileName = 'demoSpectra.h5'
	
	if fileName[-3:] != '.h5':
		fileName = fileName + '.h5'
		print('[Warning] the filename was changed to \'' + fileName + '\'')

	with h5py.File(fileName, 'w') as fw:
		fw.create_dataset('f', data=freq)
		fw.create_dataset('c', data=velo)
		fw.create_dataset('amp', data=spectrum)
		
def curveInterp(curve, freqSeries=[]):
	curve = np.array(curve)
	curve = curve[np.argsort(curve[:,-1])]
	
	if curve.shape[1] == 2 or curve.shape[1] == 4:
		raise TypeError('mode value is NO FOUND, plesase use the function \'dispernet.modeSeparation\' to divide the curve to different modes.')
		
		
	if freqSeries == []:
		freqSeries = np.linspace(0,10,101)
	
	outputCurve = []
	for mode in range(int(max(curve[:,-1]))+1):
		curveInMode = curve[curve[:,-1] == mode]
		curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
		
		if len(curveInMode) < 2:
			print('[Warning] dispersion curve at mode ' + str(mode) + ' has less than 2 points, which can not be interpolated')
			outputCurve = np.vstack([outputCurve, curveInMode])
			continue
		
		fMax = max(curveInMode[:,0])
		fMin = min(curveInMode[:,0])
		
		freqSeriesPart = freqSeries[freqSeries <= fMax]
		freqSeriesPart = freqSeriesPart[freqSeriesPart >= fMin]
		
		if len(freqSeriesPart) == 0:
			print('[Warning] points in dispersion curve at mode ' + str(mode) + ' are too close, which can not be interpolated')
			outputCurve = np.vstack([outputCurve, curveInMode])
			continue
		
		veloInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,1])
		
		
		
		if curve.shape[1] > 3:
			veloMaxInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,2])
			veloMinInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,3])
			if mode == 0:
				outputCurve = np.vstack([freqSeriesPart, veloInterp, veloMaxInterp, veloMinInterp, mode*np.ones(len(freqSeriesPart))]).T
			else:
				outputCurve = np.vstack([outputCurve, np.vstack([freqSeriesPart, veloInterp, veloMaxInterp, veloMinInterp, mode*np.ones(len(freqSeriesPart))]).T])
		else:
			if mode == 0:
				outputCurve = np.vstack([freqSeriesPart, veloInterp, mode*np.ones(len(freqSeriesPart))]).T
			else:
				outputCurve = np.vstack([outputCurve,np.vstack([freqSeriesPart, veloInterp, mode*np.ones(len(freqSeriesPart))]).T])
		
	return np.squeeze(outputCurve)
	
def curveSmooth(curve, sigma=1):

	for mode in range(int(max(curve[:,2])+1)):
		curveInMode = curve[curve[:,-1] == mode]
		curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
		curve_smooth = sn.gaussian_filter1d(curveInMode[:,1],sigma)
		curve[curve[:,2]==mode,1] = curve_smooth
		curve[curve[:,2]==mode,0] = curveInMode[:,0]
		
	return curve
	
def extract(spec, threshold=0.5, freq=[0.,0.3], velo=[2000, 6000], net='noise', mode=0,freqSeries=[] ,errorbar=False, flipUp=False, searchStep=10, searchBorder=0, returnSpec=False, ind=-1, url = 'http://10.20.43.106:8514'):
	curve = pick(spec, threshold, freq, velo, net, errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, url)
	if mode > 0:
		curve = modeSeparation(curve, mode)
	else:
		curve = autoSeparation(curve)
	if len(freqSeries) > 0:
		curve = curveInterp(curve, freqSeries)
	
	return curve
		
class App(object):
	ind = 0
	cidclick = None
	cidDelete = None
	
	threshold_set = 0.5
	curve = []
	modeInClick = 0
	net_type_preset = 'noise'
	

	
	
	def __init__(self, filePath='./', curveFilePath = '', freqSeries=[], cmap='viridis', vmin=None, vmax=None):
		
		self.fig = plt.figure(figsize=[9,7])
		self.ax1=plt.subplot(111)
		plt.subplots_adjust(bottom=0.3, right=0.6)
		
		self.axUpload = plt.axes([0.755, 0.05, 0.1, 0.075])
		self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
		
		self.axModeDivide = plt.axes([0.1, 0.05, 0.1, 0.075])
		self.buttonModeDivide = Button(self.axModeDivide, 'Mode\nDivide')
		
		self.axAutoModeDivide = plt.axes([0.1, 0.15, 0.1, 0.075])
		self.buttonAutoModeDivide = Button(self.axAutoModeDivide, 'Automatic\nMode\nDivide')
		
		
		self.axAdd = plt.axes([0.25, 0.05, 0.1, 0.075])
		self.buttonAdd = Button(self.axAdd, 'Add\nPoint')
		
		self.axDelete = plt.axes([0.4, 0.05, 0.1, 0.075])
		self.buttonDelete = Button(self.axDelete, 'Delete\nPoint')

		self.axSave = plt.axes([0.55, 0.05, 0.1, 0.075])
		self.buttonSave = Button(self.axSave, 'Save\nPoint')

		self.axprev = plt.axes([0.7, 0.15, 0.1, 0.075])
		self.axnext = plt.axes([0.81, 0.15, 0.1, 0.075])
		
		self.bnext = Button(self.axnext, 'Next')
		self.bprev = Button(self.axprev, 'Previous')
		
		modeButtonXLoc = 0.64
		self.axMode4 = plt.axes([modeButtonXLoc, 0.8, 0.1, 0.05])
		self.axMode3 = plt.axes([modeButtonXLoc, 0.7, 0.1, 0.05])
		self.axMode2 = plt.axes([modeButtonXLoc, 0.6, 0.1, 0.05])
		self.axMode1 = plt.axes([modeButtonXLoc, 0.5, 0.1, 0.05])
		self.axMode0 = plt.axes([modeButtonXLoc, 0.4, 0.1, 0.05])

		self.buttonMode0 = Button(self.axMode0, 'mode 0:')
		self.buttonMode1 = Button(self.axMode1, 'mode 1:')
		self.buttonMode2 = Button(self.axMode2, 'mode 2:')
		self.buttonMode3 = Button(self.axMode3, 'mode 3:')
		self.buttonMode4 = Button(self.axMode4, 'mode 4:')
		
		self.textM4 = self.fig.text(modeButtonXLoc+0.12,0.82, '0')
		self.textM3 = self.fig.text(modeButtonXLoc+0.12,0.72, '0')
		self.textM2 = self.fig.text(modeButtonXLoc+0.12,0.62, '0')
		self.textM1 = self.fig.text(modeButtonXLoc+0.12,0.52, '0')
		self.textM0 = self.fig.text(modeButtonXLoc+0.12,0.42, '0')
		
		self.axInterp = plt.axes([0.55, 0.15, 0.1, 0.075])
		self.buttonInterp = Button(self.axInterp, 'Curve\nInterp')
		
		self.axSmooth = plt.axes([0.4, 0.15, 0.1, 0.075])
		self.buttonSmooth= Button(self.axSmooth, 'Curve\nSmooth')
		
		
		self.axth = plt.axes([0.7, 0.25, 0.2, 0.03])
		self.slth = Slider(self.axth, 'threshold', 0, 1.0, valinit=0.5)
		
		self.axNetType = plt.axes([0.82, 0.7, 0.1, 0.15])
		self.checkNetType = CheckButtons(self.axNetType, ['noise','event','noise2', 'noise3','toLB','toLB2'],[1,0,0,0,0,0])
		self.axNetType.set_title('Net Type')	
		
		
		self.buttonUpload.on_clicked(self.upload)
		self.buttonModeDivide.on_clicked(self.modeDivide)
		self.buttonAdd.on_clicked(self.add_mode_on)
		self.buttonMode0.on_clicked(self.mode0ButtonClick)
		self.buttonMode1.on_clicked(self.mode1ButtonClick)
		self.buttonMode2.on_clicked(self.mode2ButtonClick)
		self.buttonMode3.on_clicked(self.mode3ButtonClick)
		self.buttonMode4.on_clicked(self.mode4ButtonClick)
		self.bnext.on_clicked(self.next)
		self.bprev.on_clicked(self.prev)
		self.buttonSave.on_clicked(self.save)
		self.buttonDelete.on_clicked(self.deletePoint)
		self.buttonInterp.on_clicked(self.curveInterpButton)
		self.buttonSmooth.on_clicked(self.curveSmoothButton)
		self.buttonAutoModeDivide.on_clicked(self.autoDivide)
		self.slth.on_changed(self.threshold_changed)
		self.checkNetType.on_clicked(self.set_net_type)
		
		self.cmap = cmap
		self.vmin = vmin
		self.vmax = vmax
		
		self.freqSeriesForInterp = freqSeries
		
		self.fileList = self.get_file_list(filePath, end='.h5')
		self.fileList = self.natural_sort(self.fileList)
		self.filePath = filePath  + '/'
		if self.fileList == []:
			raise IOError('No *.h5 file found in the given Path, please check. \nYou can use the function dispernet.save2h5 to transfer the sptectrum to the specific *.h5 file')
			
		self.fileName = self.fileList[0]
		
		self.modeNum = 2
		
		if curveFilePath == '':
			self.curveFilePath = filePath
		else:
			self.curveFilePath = curveFilePath + '/'
		
		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
			self.spec = np.array(fr['amp'])
			
		show(self.spec,[],freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title(self.fileName)
		plt.show()

	
	def threshold_changed(self,event):
		self.threshold_set = float(event)
		
	def curveInterpButton(self,event):
		
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
			return
			
		if self.curve.shape[1] == 2:
			self.ax1.set_title('Please Devide the curve to different mode FIRST!!')
			return
		
		self.curve = curveInterp(self.curve, self.freqSeriesForInterp)
		self.ax1.cla()
		show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title('Curve Interpolation')
		plt.draw()
	
	def curveSmoothButton(self,event):
		
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
			return
			
		if self.curve.shape[1] == 2:
			self.ax1.set_title('Please Devide the curve to different mode FIRST!!')
			return
		
		self.curve = curveSmooth(self.curve, 1)
		self.ax1.cla()
		show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title('Curve Smoothed')
		plt.draw()

	
	def save(self, event):
		if self.curve == []:
			self.ax1.set_title('No curves to save yet.')
		else:
			if len(self.curve) > 1:
				if self.curve.shape[1] !=2 and  self.curve.shape[1] !=4:
					self.curve = self.curve[np.argsort(self.curve[:,-1])]
					for mode in range(int(max(self.curve[:,-1])+1)):
						curveInMode = self.curve[self.curve[:,-1] == mode]					
						self.curve[self.curve[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
			
			if self.curve.shape[1] > 2:		
				np.savetxt(self.curveFilePath + self.fileName[:-3] + 'curve.txt', self.curve, fmt='%.6f  %.6f  %i')
			else:
				np.savetxt(self.curveFilePath + self.fileName[:-3] + 'curve.txt', self.curve, fmt='%.6f  %.6f')
			self.ax1.set_title('Curve file saved. ('+str(len(self.curve)) + ' points)')
			
		plt.draw()
	
	
	def set_net_type(self,event):
		self.net_type_preset = str(event)
		netNameList = ['noise','event','noise2','noise3','toLB','toLB2']
		changedValue = np.zeros(len(netNameList))
		for ind,name in enumerate(netNameList):
			if self.net_type_preset == name:
				changedValue[ind] = 1

		self.axNetType.cla()
		self.checkNetType = CheckButtons(self.axNetType, ['noise','event','noise2', 'noise3','toLB','toLB2'],changedValue)
		self.checkNetType.on_clicked(self.set_net_type)
		self.axNetType.set_title('Net Type')
		
		plt.draw()
		
	def next(self, event):
		self.ind += 1
		
		if self.ind >= len(self.fileList):
			self.ind = 0
		
		self.fileName  = self.fileList[self.ind]
		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
			self.spec = np.array(fr['amp'])
		
		self.curve = []
		self.ax1.cla()
		show(self.spec,[],freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title(self.fileName)
		plt.draw()
			
	def prev(self, event):
		self.ind -= 1
		if self.ind <= 0:
			self.ind = len(self.fileList) - 1
			
		self.fileName  = self.fileList[self.ind]

		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
			self.spec = np.array(fr['amp'])
		self.curve = []
		
		self.ax1.cla()
		show(self.spec,[],freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title(self.fileName)
		plt.draw()

		
	def on_click(self, event):

		x = event.xdata
		y = event.ydata
		
		if x == None:
			self.fig.canvas.mpl_disconnect(self.cidclick)
			self.ax1.set_title('Add Mode off')
			plt.draw()
		else:
			if event.inaxes == self.ax1:
				newPoint = [x, y, self.modeInClick]
				if self.curve == []:
					self.curve = np.array([x, y, self.modeInClick])
				else:
					
					if self.curve.ndim > 1:
						if self.curve.shape[1] == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
					else:
						if len(self.curve) == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
						
					self.ax1.cla()
					show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
					self.ax1.set_title('Added :' + str(x) + ', '+ str(y))
					self.reflashModeLabel()
					plt.draw()
			
			
		
	def add_mode_on(self,event):
		if self.cidDelete: 
			self.fig.canvas.mpl_disconnect(self.cidDelete)
			
		self.cidclick = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
		self.ax1.set_title('Add Mode On')
		plt.draw()
	
	def deletePoint(self, event):
		if self.cidclick:
			self.fig.canvas.mpl_disconnect(self.cidclick)
			
		self.cidDelete = self.fig.canvas.mpl_connect("button_press_event", self.on_delete_event)
		self.ax1.set_title('Delete Mode On')
		plt.draw()
		
	def on_delete_event(self, event):
		x = event.xdata
		y = event.ydata
		

		if x == None:
			self.fig.canvas.mpl_disconnect(self.cidDelete)
			self.ax1.set_title('Delete Mode off')
			plt.draw()
			
		else:
			if event.inaxes == self.ax1 and self.curve != []:
				errorRangeX = (max(self.freq) - min(self.freq)) / 100
				errorRangeY = (max(self.velo) - min(self.velo)) / 100
				
				deleteList = []
				for ind, point in enumerate(self.curve):
					if (point[0] < x+errorRangeX ) and (point[0] > x-errorRangeX) and(point[1] < y+errorRangeY) and (point[1] > y-errorRangeY):
						deleteList.append(ind)
						
				
				if deleteList != []:
					self.curve = np.delete(self.curve, deleteList,0)
					
					self.ax1.cla()
					
					show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
					self.ax1.set_title('Deleted :' + str(x) + ', '+ str(y))
					self.reflashModeLabel()
					plt.draw()
			
			else:
				self.fig.canvas.mpl_disconnect(self.cidDelete)

	def reflashModeLabel(self):
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.textM0.set_text(str(len(self.curve[self.curve[:,-1]==0])))
			self.textM1.set_text(str(len(self.curve[self.curve[:,-1]==1])))
			self.textM2.set_text(str(len(self.curve[self.curve[:,-1]==2])))
			self.textM3.set_text(str(len(self.curve[self.curve[:,-1]==3])))
			self.textM4.set_text(str(len(self.curve[self.curve[:,-1]==4])))
	
	def upload(self, event):
		self.ax1.cla()
		self.curve = pick(self.spec, freq=self.freq, velo=self.velo, net=self.net_type_preset, threshold=self.threshold_set, searchStep=7)
		show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax1.set_title("DisperNet Picked: " + self.fileName)
		plt.draw()
		
	
		
	def modeDivide(self, event):
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.curve = modeSeparation(self.curve, self.modeNum)
			self.ax1.cla()
			show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
			self.ax1.set_title(self.fileName)
			self.reflashModeLabel()
			plt.draw()
			
	def autoDivide(self, event):
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.curve = autoSeparation(self.curve)
			self.modeNum = int(max(self.curve[:,2])) + 1
			self.ax1.cla()
			show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
			self.ax1.set_title('Auto Divided into '+str(self.modeNum) + ' mode(s)')
			self.reflashModeLabel()
			plt.draw()
	
	def mode0ButtonClick(self, event):
		self.modeInClick = 0
		self.modeNum = 1
		
	def mode1ButtonClick(self, event):
		self.modeInClick = 1
		self.modeNum = 2
		
	def mode2ButtonClick(self, event):
		self.modeInClick = 2
		self.modeNum = 3
		
	def mode3ButtonClick(self, event):
		self.modeInClick = 3
		self.modeNum = 4
			
	def mode4ButtonClick(self, event):
		self.modeInClick = 4
		self.modeNum =5
		
	def get_file_list(self, basis_dir="./", begin="", end=""):
		path_list = os.listdir(basis_dir)
		list_final = []
		for partial in path_list:
			if begin and end:
				if partial[:len(begin)] == begin and partial[-len(end):] == end:
					list_final.append(partial)
					
			elif end:
				if partial[-len(end):] == end:
					list_final.append(partial)
			
			elif begin:
				if partial[:len(begin)] == begin:
					list_final.append(partial)
					
			else:
				list_final.append(partial)
				
		return list_final
		
	def natural_sort(self, l): 
		convert = lambda text: int(text) if text.isdigit() else text.lower() 
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(l, key = alphanum_key)
	

def help():
	print('Welcome to DisperNet(py)!\n\nThe DisperNet(py) is a tool provides a simple and convenient way to extract the dispersion curve from the spectra automatically. If this is your first time using DisperNet, you should definitely check out the readme.md document firstly.\n\nThe  DisperNet(py) mainly contains the functions:\n\n1. save2h5(spectrum, freq, velo, fileName): save the np.array of dispersion image(spectrum) to a specific h5fs file format with amp, freq and velo information.\n2. pick(spec, threshold, freq, velo, net, errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, url): pick the dispersion curve from the arguments, which need internet connection in SUSTech. This function uploads the data to the online server and fetch the curve based on the spectrum and the extra settings in the arguments.\n3. modeSeparation(curves, modes): separate the picked dispersion curves to different modes, based on locally unsupervised classification(hierarchical clustering analyzation).\n4. show(spec,curve,freq, velo, unit, s, ax, holdon, cmap, vmin, vmax): A simple tool to plot the figure of dispersion spectrum and the curves. \n5. curveInterp(curve, freqSeries): Interpolation of the separated dispersion curve, transfer the curve a smooth and continuous series.\n6. extract(spec, threshold, freq, velo, net, mode,freqSeries ,errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, ur): the function that fuse all the functions above together, if you have decided all the parameters already, this function will promote your code :-)\n\nWe also provide a application with GUI, you can easily launch it by dispernet.App(). \n\nNOT every arguments above are necessary, instead most of them are optional. You can refer to the readme.md for more details.\n\nLastly, we list the optional network type for \'net\' argument:\n 1. noise: For abient noise data from Gaoxiong Wu\'s work. (default)\n 2. event: For earthquakes events data from Zhengbo Li\'s work\n 3. noise2\n 4. noise3\n 5. toLB: transfer learning by Long Beach City data.')

if __name__ == '__main__':
	help()
	
