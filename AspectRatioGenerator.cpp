#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono> 

#include "LandmarkPredictor.h"

void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points);

int main()
{
	// Setup Face Detector and Facial Landmark Predictor
	LP::initializePredictor();

	// open video file
	cv::VideoCapture cap("D:\\datasets\\ngantuk\\01\\0.mp4");

	if (!cap.isOpened())  // isOpened() returns true if capturing has been initialized.
	{
		std::cout << "Cannot open the video file. \n";
		return -1;
	}

	// landmaarks container
	dlib::full_object_detection landmarks;

	// current frame container
	cv::Mat currentFrame;

	// variable for storing video information
	float frameCounter = 0;

	// get frames from camera
	while (1) {

		// read current frame
		cap.read(currentFrame);

		// container for points from all areas of interest
		std::vector<cv::Point2f> coordinates;

		// Check if the program is in tracking mode
		if (!LK::isTracking()) {
			// detect landmarks
			try {
				LP::predictLandmarks(landmarks, currentFrame);

				// get points from all areas of interest
				coordinates = LP::getCoordinatesFromLandmarks(landmarks);

				// comment to disable tracking for the next frames
				LK::start(currentFrame, landmarks);
			}
			catch (int errorCode) {
				if (errorCode == 1) {
					std::cout << "no face detected" << std::endl;
					LK::setTracking(false);
				}
			}
		}
		else {
			// get points from track
			coordinates = LK::track(currentFrame);
		}

		// draw landmark points on frame
		drawLandmarks(currentFrame, coordinates);

		// Display current frame
		cv::imshow("Frame", currentFrame);

		// Press ESC on keyboard to exit
		char c = (char)cv::waitKey(1);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// closes all the frames
	cv::destroyAllWindows();

	return 0;
}

void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points) {
	for (auto const& point : points) {
		cv::circle(frame, point, 1, CV_RGB(0, 255, 0), 2);
	}
}
