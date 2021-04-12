import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import skimage
import skimage.filters as filters
import skimage.transform as transform
from skimage.util import img_as_ubyte, img_as_float
import skimage.color as color
from random import random
from io import BytesIO
import PIL
import pywt

#Hilfs Klassen
def randUnifC(low, high, params=None):
	p = np.random.uniform()
	if params is not None:
		params.append(p)
	return (high-low)*p + low

def randUnifI(low, high, params=None):
	p = np.random.uniform()
	if params is not None:
		params.append(p)
	return round((high-low)*p + low)

def randLogUniform(low, high, base=np.exp(1)):
	div = np.log(base)
	return base**np.random.uniform(np.log(low)/div,np.log(high)/div)

#Farbgenauigkeitssenkung
def CPR(img):
	name = 'CPR'
	scales = [np.asscalar(np.random.random_integers(8,200)) for x in range(3)]
	multi_channel = np.random.choice(2) == 0
	params = [multi_channel] + [s/200.0 for s in scales]
	if multi_channel:
		img= np.round(img*float(scales[0]))/float(scales[0])
	else:
		for i in range(3):
			img[:,:,i] = np.round(img[:,:,i]*float(scales[i])) / float(scales[i])
	return img, params , name

#JPEG rauschen
def JPEGR(img):
	name = 'JPEGR'
	img = img_as_float(img)
	quality = np.asscalar(np.random.random_integers(75,95))
	params = [quality/100.0]
	pil_image = PIL.Image.fromarray((img*255.0).astype(np.uint8) )
	f = BytesIO()
	pil_image.save(f, format='jpeg', quality=quality)
	jpeg_image = np.asarray( PIL.Image.open(f)).astype(np.float32) / 255.0
	return jpeg_image, params , name

#Swirl
def Swirl(img):
	name = 'Swirl'
	img = img_as_float(img)
	strength = (2.0-1)*np.random.random(1)[0] + 1
	c_x = np.random.random_integers(1, 32)
	c_y = np.random.random_integers(1, 32)
	radius = np.random.random_integers(10, 200)
	params = [strength/2.0, c_x/256.0, c_y/256.0,radius/200.0]
	img = skimage.transform.swirl(img, rotation=0,strength=strength, radius=radius, center=(c_x,c_y))
	return img, params, name

# Rauschen Hinzufügen
def NI(img):
	name = 'Ni'
	params = []
	img = img_as_float(img)
	options = ['gaussian', 'salt', 'pepper', 's&p', 'speckle']
	noise_type = np.random.choice(options, 1)[0]
	params.append(options.index(noise_type)/6.0)
	per_channel = False
	params.append( per_channel )
	if per_channel:
		for i in range(3):
			img[:,:,i] = skimage.util.random_noise(img[:,:,i], mode=noise_type )
	else:
		img = skimage.util.random_noise( img,mode=noise_type )
	return img, params, name

# Schnelle Fourier Transformation Pertubartion

def FFTP(img):
	name = 'FFTP'
	r, c, _ = img.shape
	point_factor = (1.02-0.98)*np.random.random((r,c)) +0.98
	randomized_mask = [np.random.choice(2)==0 for x in range(3)]
	keep_fraction = [(0.95-0.0)*np.random.random(1)[0] + 0.0 for x in range(3)]
	params = randomized_mask + keep_fraction
	for i in range(3):
		im_fft = fftpack.fft2(img[:,:,i])
		r, c = im_fft.shape
		if randomized_mask[i]:
			mask = np.ones(im_fft.shape[:2])> 0
			im_fft[int(r*keep_fraction[i]): int(r*(1-keep_fraction[i]))] = 0
			im_fft[:, int(c*keep_fraction[i]): int(c*(1-keep_fraction[i]))] = 0
			mask = ~mask
			mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) <keep_fraction[i])
			mask = ~mask
			im_fft = np.multiply(im_fft, mask)
		#else:
		#    im_fft[int(r*keep_fraction[i]): int(r*(1-keep_fraction[i]))] = 0
		#    im_fft[:, int(c*keep_fraction[i]): int(c*(1-keep_fraction[i]))] = 0
		im_fft = np.multiply(im_fft, point_factor)
		im_new = fftpack.ifft2(im_fft).real
		im_new = np.clip(im_new, 0, 1)
		img[:,:,i] = im_new
	return img, params, name

#Zufälliger Zoom
def ZZ(img):
	name = 'ZZ'
	h, w, _ = img.shape
	i_s = np.random.random_integers(3.0, 10.0)
	i_e = np.random.random_integers(3.0, 10.0)
	j_s = np.random.random_integers(3.0, 10.0)
	j_e = np.random.random_integers(3.0, 10.0)
	params = [i_s/15.0, i_e/15.0, j_s/15.0, j_e/15.0]
	i_e = h-i_e
	j_e = w-j_e
	img = img[i_s:i_e,j_s:j_e,:]
	img = skimage.transform.resize(img, (h, w, 3))
	return img, params, name


# Alter HSV
def AHSV(img):
	name = 'HSV'
	img = (img+1.0)*0.5
	img = color.rgb2hsv(img)
	params = []
	img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
	img[:,:,1] += randUnifC(-0.25, 0.25, params=params)
	img[:,:,2] += randUnifC(-0.25, 0.25, params=params)
	img = np.clip(img, -1, 1.0)
	img = color.hsv2rgb(img)
	#img = (img*2.0)-1.0
	img = np.clip(img, 0, 1.0)
	return img, params, name

#Alter XYZ
def AXYZ(img):
	name = 'XYZ'
	img = color.rgb2xyz(img)
	params = []
	img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
	img[:,:,1] += randUnifC(-0.05, 0.05, params=params)
	img[:,:,2] += randUnifC(-0.05, 0.05, params=params)
	img = np.clip(img, 0, 1.0)
	img = color.xyz2rgb(img)
	img = (img * 2.0) - 1.0
	img = np.clip(img, 0, 1)
	return img, params, name

#Alter LAB
def ALAB(img):
	name = 'ALAB'
	img = (img + 1.0) * 0.5
	img = color.rgb2lab(img)
	params = []
	img[:,:,0] += randUnifC(-5.0, 5.0, params=params)
	img[:,:,1] += randUnifC(-2.0, 2.0, params=params)
	img[:,:,2] += randUnifC(-2.0, 2.0, params=params)
	img[:,:,0] = np.clip(img[:,:,0], 0, 100.0)
	img = color.lab2rgb(img)
	img = (img * 2.0) - 1.0
	img = np.clip(img, -1.0, 1.0)
	return img, params, name

#Alter YUV
def AYUV(img):
	name = 'AYUV'
	img = img_as_float(img)
	img = (img + 1.0) * 0.5
	img = color.rgb2yuv(img)
	params = []
	img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
	img[:,:,1] += randUnifC(-0.02, 0.02, params=params)
	img[:,:,2] += randUnifC(-0.02, 0.02, params=params)
	img[:,:,0] = np.clip(img[:,:,0], -1, 1.0)
	img = color.yuv2rgb(img)
	img = (img * 2.0) - 1.0
	img = np.clip(img, -1.0, 1.0)
	return img, params, name

#Histogramm Ausgleich
def HE(img):
    name = 'HE'
    #img_as_ubyte(img)
    nbins = np.random.random_integers(0,2)
    params = [ nbins/20.0 ]
    for i in range(3):
        img[:,:,i] = skimage.exposure.equalize_hist( img[:,:,i], nbins=nbins)
    #img = (img * 2.0) - 1.0
    img = np.clip(img, -1.0, 1.0)
    return img, params, name

#Adaptiver Histogramm Ausgleich
def AHE(img):
	name = 'AHE'
	
	min_size = min(img.shape[0], img.shape[1])/10
	max_size = min(img.shape[0], img.shape[1])/6
	per_channel = np.random.choice(2) == 0
	params = [ per_channel ]
	kernel_h = [ randUnifI(min_size, max_size,params=params) for x in range(3)]
	kernel_w = [ randUnifI(min_size, max_size,params=params) for x in range(3)]
	clip_lim = [randUnifC(0.01, 0.04, params=params) for x in range(3)]
	if per_channel:
		for i in range(3):
			kern = (kernel_w[i], kernel_h[i])
			img[:,:,i] =skimage.exposure.equalize_adapthist(img[:,:,i], kernel_size=kern,clip_limit=clip_lim[i])
			img = img_as_ubyte(img)
	else:
		kern = (kernel_w[0], kernel_h[0])
		img = skimage.exposure.equalize_adapthist( img,kernel_size=kern, clip_limit=clip_lim[0])
		img = img_as_ubyte(img)
	return img, params, name 

#Kontrast Dehnung
def CS(img):
	name = 'CS'
	img = img_as_float(img)
	per_channel = np.random.choice(2) == 0
	params = [ per_channel ]
	low_precentile = [ randUnifC(0.01, 0.04, params=params)for x in range(3)]
	hi_precentile = [ randUnifC(0.96, 0.99, params=params) for x in range(3)]
	if per_channel:
		for i in range(3):
			p2, p98 = np.percentile(img[:,:,i],(low_precentile[i]*100,hi_precentile[i]*100))
			img[:,:,i] =skimage.exposure.rescale_intensity(img[:,:,i], in_range=(p2, p98))
	else:
		p2, p98 = np.percentile(img, (low_precentile[0] *100, hi_precentile[0]*100))
		img = skimage.exposure.rescale_intensity( img,in_range=(p2, p98) )
	img = np.clip(img, -1.0, 1.0)
	return img, params, name

#Graustufenmix
def GSM(img):
	name = 'GSM'
	#img = img_as_ubyte(img)
	ratios = np.random.rand(3)
	ratios /= ratios.sum()
	params = [x for x in ratios]
	img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1]+ img[:,:,2] * ratios[2]
	for i in range(3):
		img[:,:,i] = img_g
	img = np.clip(img, -1.0, 1.0)
	img_as_ubyte(img)
	return img, params, name

# Teilweiser Graustufenmix
def GSPM(img):
	name = 'GSPM'
	ratios = np.random.rand(3)
	ratios/=ratios.sum()
	prop_ratios = np.random.rand(3)
	params = [x for x in ratios] + [x for x in prop_ratios]
	img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1]+ img[:,:,2] * ratios[2]
	for i in range(3):
		p = max(prop_ratios[i], 0.2)
		img[:,:,i] = img[:,:,i]*p + img_g*(1.0-p)
	return img, params, name

# 2/3 Graustufenmix
def TTGSM(img):
	name = 'TTGSM'
	img = img_as_float(img)
	params = []
	channels = [0, 1, 2]
	remove_channel = np.random.choice(3)
	channels.remove( remove_channel)
	params.append( remove_channel )
	ratios = np.random.rand(2)
	ratios/=ratios.sum()
	params.append(ratios[0])
	img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
	for i in channels:
		img[:,:,i] = img_g
	img = np.clip(img, -1.0, 1.0)
	img_as_ubyte(img)
	return img, params, name

# Ein Kanal Teilweise Grau
def OCPG(img):
	name = 'OCPG'
	params = []
	channels = [0, 1, 2]
	to_alter = np.random.choice(3)
	channels.remove(to_alter)
	params.append(to_alter)
	ratios = np.random.rand(2)
	ratios/=ratios.sum()
	params.append(ratios[0])
	img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
	p = (0.9-0.1)*np.random.random(1)[0] + 0.1
	params.append( p )
	img[:,:,to_alter] = img_g*p + img[:,:,to_alter]*(1.0-p)
	return img, params, name

#Gauß'sche Unschärfe
def GB(img):
	name = 'GB'
	img = img_as_float(img)
	if randUnifC(0, 1) > 0.5:
		sigma = [randUnifC(0.1, 3)]*3
	else:
		sigma = [randUnifC(0.1, 3), randUnifC(0.1, 3),randUnifC(0.1, 3)]
	img_as_float(img)
	img[:,:,0] = skimage.filters.gaussian(img[:,:,0],sigma=sigma[0])
	img[:,:,1] = skimage.filters.gaussian(img[:,:,1],sigma=sigma[1])
	img[:,:,2] = skimage.filters.gaussian(img[:,:,2],sigma=sigma[2])
	img = img.astype(np.uint8)
	return img, [x/3.0 for x in sigma], name

#Median Filter
def MF(img):
	name = 'MF'
	img = img/255
	if randUnifC(0, 1) > 0.5:
		radius = [randUnifI(2, 5)]*3
	else:
		img_as_ubyte(img)
		radius = [randUnifI(2, 5), randUnifI(2, 5),randUnifI(2, 5)]
	for i in range(3):
		mask = skimage.morphology.disk(radius[i])
		img[:,:,i] = skimage.filters.rank.median(img[:,:,i], mask) / 255.0
	return img, [x/5.0 for x in radius], name

# Durchschnitts Filter
def MEANF(img):
	img_as_ubyte(img)
	name = 'MEANF'
	#img = img  /255 
	if randUnifC(0, 1) > 0.5:
		radius = [randUnifI(2, 3)]*3
	else:
		radius = [randUnifI(2, 3), randUnifI(2, 3),randUnifI(2, 3)]
	for i in range(2):
		mask = skimage.morphology.disk(radius[i])
		img[:,:,i] = skimage.filters.rank.mean(img[:,:,i],mask)
	img = np.asarray(img)
	return img, [x/3.0 for x in radius], name

# Durchschnitts Biliterale Filterung
def MBF(img):
	name = 'MBF'
	img = img  /255 
	img = img_as_float(img)
	params = []
	radius = []
	ss = []
	for i in range(3):
		radius.append( randUnifI(2, 20, params=params) )
		ss.append( randUnifI(5, 20, params=params) )
		ss.append( randUnifI(5, 20, params=params) )
	for i in range(3):
		mask = skimage.morphology.disk(radius[i])
		img[:,:,i] = skimage.filters.rank.mean_bilateral(img[:,:,i], mask, s0=ss[i], s1=ss[3+i])/255.0 
	img = np.clip(img, -1.0, 1.0)
	return img, params, name

#Chambolle Entrauschen
def CD(img):
	img = img_as_float(img)
	name = 'CD'
	params = []
	weight = (0.25-0.05)*np.random.random(1)[0] + 0.05
	params.append( weight )
	multi_channel = np.random.choice(2) == 0
	params.append( multi_channel )
	img = skimage.restoration.denoise_tv_chambolle( img,weight=weight, multichannel=multi_channel)
	img = np.clip(img, -1.0, 1.0)
	return img, params, name

#Wavlet Entrauschen
class WD():
	def __init__(self):
		self.wavelets = pywt.wavelist('db')
	def wd(self, img):
		name = 'WD'
		img= img_as_ubyte(img)
		convert2ycbcr = np.random.choice(2) == 0
		wavelet = np.random.choice(self.wavelets)
		mode_ = np.random.choice(["soft", "hard"])
		denoise_kwargs = dict(multichannel=True,convert2ycbcr=convert2ycbcr, wavelet=wavelet,mode=mode_)
		max_shifts = np.random.choice([0, 1])
		params = [convert2ycbcr, self.wavelets.index(wavelet)/float(len(self.wavelets)), max_shifts/5.0,(mode_=="soft")]
		img = skimage.restoration.cycle_spin(img,func=skimage.restoration.denoise_wavelet,max_shifts=max_shifts, func_kw=denoise_kwargs,multichannel=True, num_workers=1)
		return img, params, name

#Nicht Lokales Durchschnitts Entrauschen
def NLMD(img):
	name = 'NLMD'
	img_as_float(img)
	h_1 = randUnifC(0, 1)
	params = [h_1]
	sigma_est = np.mean(skimage.restoration.estimate_sigma(img,multichannel=True) )
	h = (1.15-0.6)*sigma_est*h_1 + 0.6*sigma_est
	multi_channel = np.random.choice(2) == 0
	params.append( multi_channel )
	fast_mode = True
	patch_size = np.random.random_integers(5, 7)
	params.append(patch_size)
	patch_distance = np.random.random_integers(6, 11)
	params.append(patch_distance)
	if multi_channel:
		img = skimage.restoration.denoise_nl_means( img,h=h, patch_size=patch_size,patch_distance=patch_distance,fast_mode=fast_mode , preserve_range=True)
		img_as_ubyte(img)
	else:
		for i in range(3):
			sigma_est = np.mean(skimage.restoration.estimate_sigma(img[:,:,i], multichannel=True ) )
			h = (1.15-0.6)*sigma_est*params[i] +0.6*sigma_est
			img[:,:,i] =skimage.restoration.denoise_nl_means(img[:,:,i], h=h, patch_size=patch_size,patch_distance=patch_distance,fast_mode=fast_mode )
			img_as_ubyte(img)
	return img, params, name

#Lege gruppen an 

def getRaGrOSev(img):
	num = np.random.randint(1,5)
	if num ==1:
		img , params, name  = AHSV(img)
	elif(num ==2):
		img , params, name  = AXYZ(img)
	elif(num ==3):
		img , params, name  = ALAB(img)
	elif(num ==4):
		img , params, name = AYUV(img)
	return img , params, name 

def getRaGrOEig(img):
	num = np.random.randint(1,3)
	#if num == 1:
	 #   img , params, name  = HE(img)
	#if(num ==2):
	#	img , params, name  = AHE(img)
	#elif(num ==3):
	img , params, name  = CS(img)
	return img , params, name 

def getRaGrONin(img):
	num = np.random.randint(1,4)
	if num ==1:
		img , params, name  = GSM(img)
	#elif(num ==2):
	#	img , params, name  = GSPM(img)
	elif(num ==2):
		img , params, name  = TTGSM(img)
	elif(num ==3):
		img , params, name = OCPG(img)
	return img , params, name 

def getRaGrOTen(img):
	num = np.random.randint(1,5)
	#if num ==1:
	#    img , params, name = GB(img)
	if(num ==1):
		img , params, name = MF(img)
	#elif(num ==2):
	#	img , params, name  = MEANF(img)
	elif(num ==2):
		img , params, name  = MBF(img)
	elif(num ==3):
		img , params, name  = CD(img)
	elif(num ==4):
		x= WD()
		img , params, name  = x.wd(img)
	#elif(num ==6):
	   # img , params, name  = NLMD(img)
	return img , params, name 

def Transform( img , dataset ,numTrans= 5):
	if(dataset == 'F-MNSIT'):
		img = color.grey2rgb(img)
	i=1
	params =[]
	alreadyUsed = []
	GroupOfTransforms = []
	NamesOfTransforms = []
	if numTrans > 10:
		numTrans =5
		print("zuviele Transformationen , es wurde 5 gewählt")
	while i <=numTrans:
		transNum = np.random.randint(1,11)
		if transNum == 1:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name = CPR(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("CPR")
				NamesOfTransforms.append(name)
			else:
				i= i-1
		elif transNum == 2:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = JPEGR(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("JPEGR")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 3:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = Swirl(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Swirl")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 4:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = NI(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Noise Injektion")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 5:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = FFTP(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("FFTP")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 6:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = ZZ(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Zoom Gruppe")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 7:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = getRaGrOSev(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Farbraumgruppe")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 8:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = getRaGrOEig(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Kontrastgruppe")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 9:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = getRaGrONin(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Graustufengruppe")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		elif transNum == 10:
			if(transNum not in alreadyUsed):
				img = img_as_float(img)
				img , params, name  = getRaGrOTen(img)
				alreadyUsed.append(transNum)
				GroupOfTransforms.append("Entrauschengruppe")
				NamesOfTransforms.append(name)
			else:
				i = i-1
		else:
			print("error")
		i = i+1
	print(GroupOfTransforms)
	print(NamesOfTransforms)
	img = np.clip(img,-1,1)
	if(dataset == 'F-MNSIT'):
		img = color.rgb2grey(img)
	return img_as_ubyte(img,True)

def create_BaRT_Set(numTrans,dataset):

	if(dataset=='CIFAR10'):
		(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	elif(dataset == 'F-MNIST'):
		(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

	transformedIm = []
	transformedValIm = []

	for image in train_images:
		image = Transform(image,dataset,numTrans)
		transformedIm.append(image)
	for image in test_images:
		image = Transform(image,dataset,numTrans)		
		transformedValIm.append(image)
	return transformedIm, transformedValIm

