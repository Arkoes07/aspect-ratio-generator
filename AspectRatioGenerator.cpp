#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono> 

#include "LandmarkPredictor.h"

namespace fs = std::filesystem;

// data path
const std::string dataPath = "D:\\datasets\\ngantuk\\data";
// output path
const std::string outPath = "D:\\datasets\\ngantuk\\csv";

void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points);
std::vector<cv::Point2f> getPartCoordinates(std::vector<cv::Point2f> allCoordinates, int partIdx);
float pointEuclideanDist(cv::Point2f p, cv::Point2f q);
float eyeAspectRatio(std::vector<cv::Point2f> coordinates);
float mouthAspectRatio(std::vector<cv::Point2f> coordinates);

int main()
{
	// Setup Face Detector and Facial Landmark Predictor
	LP::initializePredictor();

	// list of folders
	std::string folders[] = { "01", "05", "06", "07", "08" };
	int folderLen = sizeof(folders) / sizeof(folders[0]);

	// loop through the folders
	for (int idx = 0; idx < folderLen; ++idx) {
		// get folder number
		std::string subject = folders[idx];

		// create a file for output
		std::ofstream file;
		file.open(outPath + "\\" + subject + ".csv", std::ios_base::app);

		for (const auto& entry : fs::directory_iterator(dataPath + "\\" + subject)) {

			// get file name (class)
			std::string classNumber = entry.path().stem().string();

			// Create a VideoCapture object and open the input file
			cv::VideoCapture cap(entry.path().string());

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
			float frameCounter = 1;

			// get frames from camera
			while (1) {

				// read current frame
				cap.read(currentFrame);

				// check if the video has finished
				if (currentFrame.empty()) {
					LK::setTracking(false);
					break;
				}

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
						//LK::start(currentFrame, landmarks);
					}
					catch (int errorCode) {
						if (errorCode == 1) {
							LK::setTracking(false);
						}
					}
				}
				else {
					// get points from track
					coordinates = LK::track(currentFrame);
				}

				// write output
				file << subject << ";" << classNumber << ";" << frameCounter;

				// get aspect ratio 		
				if (!coordinates.empty()) {
					// container
					float aspectRatio[3];
					// check eye closed
					aspectRatio[0] = eyeAspectRatio(getPartCoordinates(coordinates, 0));
					aspectRatio[1] = eyeAspectRatio(getPartCoordinates(coordinates, 1));
					aspectRatio[2] = mouthAspectRatio(getPartCoordinates(coordinates, 2));
					// print
					file << ";" << aspectRatio[0] << ";" << aspectRatio[1] << ";" << aspectRatio[2];
				}
				else {
					file << ";;;";
				}

				// end row
				file << std::endl;

				// increase frame counter
				frameCounter++;
			}

			// When everything done, release the video capture object
			cap.release();
		}
		file.close();

		std::string&& finish = "process finished for folder : " + subject;
		std::cout << finish << std::endl;
	}
	// closes all the frames
	cv::destroyAllWindows();

	return 0;
}

void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points) {
	for (auto const& point : points) {
		cv::circle(frame, point, 1, CV_RGB(0, 255, 0), 2);
	}
}

std::vector<cv::Point2f> getPartCoordinates(std::vector<cv::Point2f> allCoordinates, int partIdx) {
	const int PART[3][2] = { {0,5}, {6,11}, {12,19} };
	std::vector<cv::Point2f> coordinates;
	for (int idx = PART[partIdx][0]; idx <= PART[partIdx][1]; ++idx) {
		coordinates.emplace_back(std::forward<cv::Point2f>(allCoordinates[idx]));
	}
	return coordinates;
}

float pointEuclideanDist(cv::Point2f p, cv::Point2f q) {
	float a = q.x - p.x;
	float b = q.y - p.y;
	return std::sqrt(a * a + b * b);
}

float eyeAspectRatio(std::vector<cv::Point2f> coordinates) {
	// compute the euclidean distances between the two sets of vertical eye landmarks(x, y) - coordinates
	float a = pointEuclideanDist(coordinates[1], coordinates[5]);
	float b = pointEuclideanDist(coordinates[2], coordinates[4]);
	// compute the euclidean distance between the horizontal eye landmark(x, y) - coordinates
	float c = pointEuclideanDist(coordinates[0], coordinates[3]);
	// compute eye aspect ratio
	return (a + b) / (2 * c);
}

float mouthAspectRatio(std::vector<cv::Point2f> coordinates) {
	// compute the euclidean distances between the three sets of vertical mouth landmarks (x, y)-coordinates
	float a = pointEuclideanDist(coordinates[1], coordinates[7]);
	float b = pointEuclideanDist(coordinates[2], coordinates[6]);
	float c = pointEuclideanDist(coordinates[3], coordinates[5]);
	// compute the euclidean distance between the horizontal mouth landmark (x, y)-coordinates
	float d = pointEuclideanDist(coordinates[0], coordinates[4]);
	// compute mouth aspect ratio
	return (a + b + c) / (3 * d);
}
