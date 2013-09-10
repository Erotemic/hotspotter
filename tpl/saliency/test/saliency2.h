// 
// File:   saliency.h
// Author: Sebastian Montabone
// Email: samontab@puc.cl
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

class Saliency
{
public:
	Saliency();
	void calcIntensityChannel(IplImage *src, IplImage *dst);
};

#endif	/* _saliency_H */

