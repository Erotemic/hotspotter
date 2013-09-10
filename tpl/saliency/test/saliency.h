// 
// File:   saliency.h
// Author: Sebastian Montabone
//
// Created on September 24, 2008, 2:57 PM
//
// Fine-grained Saliency library (FGS). 
// This library calculates fine grained saliency in real time using integral images.
// It requires OpenCV.
//

#ifndef _saliency_H
#define	_saliency_H
#include <cv.h>



class __declspec(dllexport) Saliency
{
public:
	Saliency();
	void calcIntensityChannel(IplImage *src, IplImage *dst);

private:
	void copyImage(IplImage *src, IplImage *dst);	
	void getIntensityScaled(IplImage * integralImage, IplImage * gray, IplImage *saliencyOn, IplImage *saliencyOff, int neighborhood);
	float getMean(IplImage * srcArg, CvPoint PixArg, int neighbourhood, int centerVal);
	void mixScales(IplImage **saliencyOn, IplImage *intensityOn, IplImage **saliencyOff, IplImage *intensityOff, const int numScales);
	void mixOnOff(IplImage *intensityOn, IplImage *intensityOff, IplImage *intensity);
	void getIntensity(IplImage *srcArg, IplImage *dstArg,  IplImage *dstOnArg,  IplImage *dstOffArg, bool generateOnOff);
};

#endif	/* _saliency_H */

