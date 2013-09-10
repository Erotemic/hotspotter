/*********************************************
*      Fine Grained Saliency for OpenCV      *
*         ©2008 Sebastian Montabone          *
*              samontab@puc.cl               *
*********************************************/

#include <stdio.h>
#include "saliency.h"
#include <highgui.h>

bool running = true;
void readInput();

int main(int argc, char** argv)
{
	char filename[200];
	IplImage *srcImg, *dstImg;
	char *win1 = "Source"; 
	char *win2 = "Saliency"; 

	if(argc > 1)
		sprintf(filename, "%s", argv[1]);
	else
		sprintf(filename, "lena.jpg");

	srcImg = cvLoadImage(filename);
	dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);


	//Create windows
	cvNamedWindow(win1);
	cvMoveWindow(win1, 0, 100);
	cvNamedWindow(win2);
	cvMoveWindow(win2, srcImg->width + 10, 100);



	Saliency *saliency = new Saliency;
	
	double t = (double)cvGetTickCount();
	//get saliency
	saliency->calcIntensityChannel(srcImg, dstImg);

	cvSaveImage("saliency.jpg", dstImg);
//	IplImage * im2 = cvCreateImage(cvSize(srcImg->width/8, srcImg->height/8), 8, 1);
//	cvResize(dstImg, im2);
//	cvResize(im2, dstImg);
//	cvReleaseImage(&im2);

	t = (double)cvGetTickCount() - t;
	
	printf( "Image processed in %gms\n", t/((double)cvGetTickFrequency()*1000.) );
	printf( "Press ESC to exit.\n");

	cvShowImage(win1, srcImg);
	cvShowImage(win2, dstImg);

	while(running)
	{
		//read user input
		readInput();
	}


	// release resources
	cvReleaseImage(&srcImg);	
	cvReleaseImage(&dstImg);
	cvDestroyAllWindows();

	return 0;
}


void readInput()
{
	char keyPressed = cvWaitKey(10);

	switch (keyPressed)
	{
	case 27: //code for ESC key
		{
			running = false;
			break;
		}
	}
}
