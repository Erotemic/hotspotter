// 
// File:   saliency.cpp
// Author: Sebastian Montabone
//
// Created on September 24, 2008, 2:58 PM
//
// Fine-grained Saliency library (FGS). 
// This library calculates fine grained saliency in real time using integral images.
// It requires OpenCV.
//
#include "saliency.h"


Saliency::Saliency()
{
 
}






void Saliency::copyImage(IplImage * srcArg, IplImage *dstArg)
{
	if( (srcArg->width != dstArg->width) || (srcArg->height != dstArg->height) )
	{
		//("Error: Source and Destiny images must have the same number of channels.\n");
		return;
	}

    int x, y;
    
    if(dstArg->nChannels == 1)
    {
        for(y=0;y<dstArg->height;y++)
        {
            for(x=0;x<dstArg->width;x++)
            {
                ((uchar*)(dstArg->imageData + dstArg->widthStep*y))[x] = ((uchar*)(srcArg->imageData + srcArg->widthStep*y))[x];
            }
        }
    }
    else if(dstArg->nChannels == 3)
    {
     for(y=0;y<dstArg->height;y++)
        {
            for(x=0;x<dstArg->width;x++)  
            {
                ((uchar*)(dstArg->imageData + dstArg->widthStep*y))[x*3] = ((uchar*)(srcArg->imageData + srcArg->widthStep*y))[x*3];
                ((uchar*)(dstArg->imageData + dstArg->widthStep*y))[x*3+1] = ((uchar*)(srcArg->imageData + srcArg->widthStep*y))[x*3+1];
                ((uchar*)(dstArg->imageData + dstArg->widthStep*y))[x*3+2] = ((uchar*)(srcArg->imageData + srcArg->widthStep*y))[x*3+2];
            }
        }   
    }        
}


void Saliency::calcIntensityChannel(IplImage *srcArg, IplImage *dstArg)
{
	if(dstArg->nChannels > 1)
	{
		//("Error: Destiny image must have only one channel.\n");
		return;
	}
	  const int numScales = 6;
      IplImage * intensityScaledOn[numScales];
      IplImage * intensityScaledOff[numScales];
	  IplImage * gray = cvCreateImage( cvSize(srcArg->width, srcArg->height), 8, 1 );
	  IplImage * integralImage = cvCreateImage( cvSize(gray->width+1, gray->height+1), IPL_DEPTH_32S, 1 );    
      IplImage * intensity = cvCreateImage( cvGetSize(gray), 8, 1 );
      IplImage * intensityOn = cvCreateImage( cvGetSize(gray), 8, 1 );
      IplImage * intensityOff = cvCreateImage( cvGetSize(gray), 8, 1 );

	  int i;
	  int neighborhood;
	  int neighborhoods[] = {3*4, 3*4*2, 3*4*2*2, 7*4, 7*4*2, 7*4*2*2};

	  for(i=0; i<numScales; i++)
	  {
		intensityScaledOn[i] = cvCreateImage( cvGetSize(gray), 8, 1 );
		intensityScaledOff[i] = cvCreateImage( cvGetSize(gray), 8, 1 );
	  }
           
 

	  // Prepare the input image: put it into a grayscale image.
      if(srcArg->nChannels==3)
      {
		cvCvtColor( srcArg, gray, CV_BGR2GRAY );
      }
      else if(srcArg->nChannels==1)
      {      
        copyImage(srcArg, gray);
      }


	// smooth pixels at least twice, as done by Frintrop and Itti
    cvSmooth(gray, gray, CV_GAUSSIAN, 3, 3);
    cvSmooth(gray, gray, CV_GAUSSIAN, 3, 3);
      

	// Calculate integral image, only once.
    cvIntegral(gray, integralImage);    
    
    
      
                        
for(i=0; i< numScales; i++)
{
      neighborhood = neighborhoods[i] ;
      getIntensityScaled(integralImage, gray, intensityScaledOn[i], intensityScaledOff[i], neighborhood);
}     


	  mixScales(intensityScaledOn, intensityOn, intensityScaledOff, intensityOff, numScales);

      


		mixOnOff(intensityOn, intensityOff, intensity);      
		copyImage(intensity, dstArg);
		
	
      
      cvReleaseImage(&gray);

      cvReleaseImage(&integralImage);
      cvReleaseImage(&intensity);
      cvReleaseImage(&intensityOn);
      cvReleaseImage(&intensityOff);
      

	  for(i = 0; i< numScales ; i++)
	  {
			cvReleaseImage(&(intensityScaledOn[i]));
			cvReleaseImage(&(intensityScaledOff[i]));
	  }
}






void Saliency::getIntensityScaled(IplImage * integralImage, IplImage * gray, IplImage *intensityScaledOn, IplImage *intensityScaledOff, int neighborhood)
{
    float value, meanOn, meanOff;
    CvPoint point;
    int x,y;
    cvZero(intensityScaledOn);
    cvZero(intensityScaledOff);
        
    
    for(y = 0; y < gray->height; y++)
    {
        for(x = 0; x < gray->width; x++)
        {
            point.x = x;
            point.y = y;        
            value = getMean(integralImage, point, neighborhood, ((uchar*)(gray->imageData + gray->widthStep*y))[x]);
            
            meanOn = ((uchar*)(gray->imageData + gray->widthStep*y))[x] - value;
            meanOff = value - ((uchar*)(gray->imageData + gray->widthStep*y))[x];

            if(meanOn > 0)
                ((uchar*)(intensityScaledOn->imageData + intensityScaledOn->widthStep*y))[x] = (uchar)meanOn;    
            else            
                ((uchar*)(intensityScaledOn->imageData + intensityScaledOn->widthStep*y))[x] = 0;    
                
            if(meanOff > 0)            
                ((uchar*)(intensityScaledOff->imageData + intensityScaledOff->widthStep*y))[x] = (uchar)meanOff;    
            else
                ((uchar*)(intensityScaledOff->imageData + intensityScaledOff->widthStep*y))[x] = 0;                         
        }
    }
      

    
}





float Saliency::getMean(IplImage * srcArg, CvPoint PixArg, int neighbourhood, int centerVal)
{
    CvPoint P1, P2;
    float value;
    
    P1.x = PixArg.x - neighbourhood + 1;
    P1.y = PixArg.y - neighbourhood + 1;
    P2.x = PixArg.x + neighbourhood + 1;
    P2.y = PixArg.y + neighbourhood + 1;
    
    if(P1.x < 0)
        P1.x = 0;
    else if(P1.x > srcArg->width - 1)
        P1.x = srcArg->width - 1;
    if(P2.x < 0)
        P2.x = 0;
    else if(P2.x > srcArg->width - 1)
        P2.x = srcArg->width - 1;
    if(P1.y < 0)
        P1.y = 0;
    else if(P1.y > srcArg->height - 1)
        P1.y = srcArg->height - 1;
    if(P2.y < 0)
        P2.y = 0;
    else if(P2.y > srcArg->height - 1)
        P2.y = srcArg->height - 1;
    
    // we use the integral image to compute fast features
    value = (float) (
            ((int*)(srcArg->imageData + srcArg->widthStep*P2.y))[P2.x] +
            ((int*)(srcArg->imageData + srcArg->widthStep*P1.y))[P1.x] -
            ((int*)(srcArg->imageData + srcArg->widthStep*P2.y))[P1.x] -
            ((int*)(srcArg->imageData + srcArg->widthStep*P1.y))[P2.x] 
	);

    value = (value - centerVal)/  (( (P2.x - P1.x) * (P2.y - P1.y))-1)  ;
    
    return value;
}





void Saliency::mixScales(IplImage **intensityScaledOn, IplImage *intensityOn, IplImage **intensityScaledOff, IplImage *intensityOff, const int numScales)
{
	int i=0, j=0, x, y;
	int width = intensityScaledOn[0]->width;
	int height = intensityScaledOn[0]->height;
	int maxValOn = 0, currValOn=0;
	int maxValOff = 0, currValOff=0;
	int maxValSumOff = 0, maxValSumOn=0;
	IplImage *mixedValuesOn = cvCreateImage(cvSize(width, height), IPL_DEPTH_16U, 1);
	IplImage *mixedValuesOff = cvCreateImage(cvSize(width, height), IPL_DEPTH_16U, 1);

	cvZero(mixedValuesOn);	
	cvZero(mixedValuesOff);	

	for(i=0;i<numScales;i++)
	{
		for(y=0;y<height;y++)
			for(x=0;x<width;x++)
			{
					  currValOn = ((uchar*)(intensityScaledOn[i]->imageData + intensityScaledOn[i]->widthStep*y))[x];
					  if(currValOn > maxValOn)
						  maxValOn = currValOn;

					  currValOff = ((uchar*)(intensityScaledOff[i]->imageData + intensityScaledOff[i]->widthStep*y))[x];
					  if(currValOff > maxValOff)
						  maxValOff = currValOff;

					  ((short*)(mixedValuesOn->imageData + mixedValuesOn->widthStep*y))[x] += currValOn;
					  ((short*)(mixedValuesOff->imageData + mixedValuesOff->widthStep*y))[x] += currValOff;
			}
	}

		for(y=0;y<height;y++)
			for(x=0;x<width;x++)
			{
				currValOn = ((short*)(mixedValuesOn->imageData + mixedValuesOn->widthStep*y))[x];
				currValOff = ((short*)(mixedValuesOff->imageData + mixedValuesOff->widthStep*y))[x];
					  if(currValOff > maxValSumOff)
						  maxValSumOff = currValOff;
					  if(currValOn > maxValSumOn)
						  maxValSumOn = currValOn;
			}
	

		for(y=0;y<height;y++)
			for(x=0;x<width;x++)
			{
				((uchar*)(intensityOn->imageData + intensityOn->widthStep*y))[x] = (uchar)(255.*((float)((short*)(mixedValuesOn->imageData + mixedValuesOn->widthStep*y))[x] / (float)maxValSumOn));
				((uchar*)(intensityOff->imageData + intensityOff->widthStep*y))[x] = (uchar)(255.*((float)((short*)(mixedValuesOff->imageData + mixedValuesOff->widthStep*y))[x] / (float)maxValSumOff));
			}
			

cvReleaseImage(&mixedValuesOn);
cvReleaseImage(&mixedValuesOff);

}


void Saliency::mixOnOff(IplImage *intensityOn, IplImage *intensityOff, IplImage *intensityArg)
{
	int x,y;
	int width = intensityOn->width;
	int height= intensityOn->height;
int maxVal=0;

int currValOn, currValOff, maxValSumOff, maxValSumOn;

	IplImage *intensity = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);


maxValSumOff = 0;
maxValSumOn = 0;

		for(y=0;y<height;y++)
			for(x=0;x<width;x++)
			{
				currValOn = ((uchar*)(intensityOn->imageData + intensityOn->widthStep*y))[x];
				currValOff = ((uchar*)(intensityOff->imageData + intensityOff->widthStep*y))[x];
					  if(currValOff > maxValSumOff)
						  maxValSumOff = currValOff;
					  if(currValOn > maxValSumOn)
						  maxValSumOn = currValOn;
			}
	
if(maxValSumOn > maxValSumOff)
maxVal = maxValSumOn;
else
maxVal = maxValSumOff;



		for(y=0;y<height;y++)
			for(x=0;x<width;x++)
			{
				((uchar*)(intensity->imageData + intensity->widthStep*y))[x] = (uchar)(255.*((float)(((uchar*)(intensityOn->imageData + intensityOn->widthStep*y))[x] + ((uchar*)(intensityOff->imageData + intensityOff->widthStep*y))[x]) / (float)maxVal));
			}

	copyImage(intensity, intensityArg);
	cvReleaseImage(&intensity);
}
