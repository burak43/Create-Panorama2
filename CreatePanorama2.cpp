/*
 * CreatePanorama2.cpp
 *
 *  Created on: Jun 8, 2016
 *      Author: Burak Mandira
 */

// Program to illustrate SIFT keypoint and descriptor extraction, and matching using brute force
// Author: Samarth Manoj Brahmbhatt, University of Pennsylvania

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>
#include <ctime>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

int main()
{
	clock_t start = clock();

	Mat train = imread( "1.jpg"), train_g;
	cvtColor(train, train_g, CV_BGR2GRAY);

	//detect SIFT keypoints and extract descriptors in the train image
	vector<KeyPoint> train_kp;
	Mat train_desc;

	/* SURF detects more feature than SHIFT (making 5. parameter upright = true makes it fast since checking rotation is eliminated)
	 * SURF detector & SIFT extractor give more accuracy but takes much more time
	 * */

//	SiftFeatureDetector featureDetector;
	SurfFeatureDetector featureDetector;

	featureDetector.detect(train_g, train_kp);

//	SiftDescriptorExtractor featureExtractor;
	SurfDescriptorExtractor featureExtractor(500, 4, 2, true, true);
	featureExtractor.compute(train_g, train_kp, train_desc);

	// Brute Force based descriptor matcher object
	// BFMatcher matcher;
	FlannBasedMatcher matcher;
	vector<Mat> train_desc_collection(1, train_desc);

	matcher.add(train_desc_collection);
//	no need when the matcher is BF
//	matcher.train();

	// Test image
	Mat test = imread( "2.jpg"), test_g;
	cvtColor(test, test_g, CV_BGR2GRAY);

	//detect SIFT keypoints and extract descriptors in the test image
	vector<KeyPoint> test_kp;
	Mat test_desc;

	featureDetector.detect(test_g, test_kp);
	featureExtractor.compute(test_g, test_kp, test_desc);

	// match train and test descriptors, getting 2 nearest neighbors for all test descriptors
	vector<vector<DMatch> > matches;
	matcher.knnMatch(test_desc, matches, 2);

	// filter for good matches according to Lowe's algorithm
	vector<DMatch> good_matches;
	for(size_t i = 0; i < matches.size(); i++)
	{
		if(matches[i][0].distance < 0.5 * matches[i][1].distance)
			good_matches.push_back(matches[i][0]);
		cout << i << endl;
	}

	Mat img_show;
//	drawMatches(test, test_kp, train, train_kp, good_matches, img_show, Scalar(0,255,0), Scalar(0,0,255));
//
//	resize(img_show, img_show, Size(1360, 710));

	vector<Point2f> p1, p2;
	for( size_t i = 0; i < good_matches.size(); ++i)
	{
		p1.push_back( test_kp[good_matches[i].queryIdx].pt);
		p2.push_back( train_kp[good_matches[i].trainIdx].pt);
	}

	Mat transformationMatrix = findHomography( p1, p2, CV_RANSAC, 2);

	warpPerspective( test, img_show, transformationMatrix, Size(test.cols + train.cols, test.rows));
	Mat half( img_show, Rect(0,0,train.cols,train.rows));
	train.copyTo(half);

	clock_t stop = clock();
	cout << "Elapsed time: " << (stop - start) / (double)CLOCKS_PER_SEC*1000 << endl;

	resize(img_show, img_show, Size(1360, 710));
	imshow("Matches", img_show);

	while(char(waitKey(0)) != 'q');

	return 0;
}



