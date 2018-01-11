#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <iostream>
#include <unordered_set>
#include <algorithm>


const std::string inBase = "data\\full\\adjusted\\PCO_Tracking__20171130_185356_";
const std::string outBase = "data\\processed\\";

int defaultThreshold = 50;
int maxThreshold = 400;

cv::Mat prevImg, currImg, colourImg, flow;
cv::Rect ROI;
std::vector<cv::Point> contour;


static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color) {
	for (int y = 0; y < cflowmap.rows; y += step) {
		for (int x = 0; x < cflowmap.cols; x += step) {
			const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x) * 10;
			line(cflowmap, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			cv::circle(cflowmap, cv::Point(x, y), 2, color, -1);
		}
	}
}

static void onTrackbar(int pos, void* ptr) {
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat croppedROI = prevImg(ROI).clone();
	cv::threshold(croppedROI, croppedROI, pos, 0, cv::THRESH_TOZERO);
	cv::findContours(croppedROI, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	contour = *std::max_element(contours.begin(), contours.end(), [](std::vector<cv::Point> c1, std::vector<cv::Point> c2) -> bool { return c1.size() < c2.size(); });
	cv::approxPolyDP(contour, contour, 4, true);
	cv::cvtColor(prevImg, colourImg, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < contour.size(); i++) {
		contour[i].x += ROI.x;
		contour[i].y += ROI.y;
	}
	cv::drawContours(colourImg, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 255));
	cv::imshow("Configure ROI", colourImg);
}

static void opticalFlow() {
	prevImg = cv::imread(inBase + "14000_700_Green.tiff.tif", cv::IMREAD_GRAYSCALE); // TODO: try UNCHANGED

	ROI = cv::selectROI("Select ROI", prevImg);
	cv::destroyWindow("Select ROI");

	cv::namedWindow("Configure ROI", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Threshold Trackbar", "Configure ROI", &defaultThreshold, maxThreshold, onTrackbar);
	onTrackbar(50, NULL);
	cv::waitKey();
	cv::destroyWindow("Configure ROI");

	cv::cvtColor(prevImg, colourImg, cv::COLOR_GRAY2BGR);
	cv::drawContours(colourImg, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 255));
	cv::imwrite(outBase + "700.tiff", colourImg);

	for (int idx = 701; idx <= 910; idx++) {
		currImg = cv::imread(inBase + std::to_string(idx * 20) + "_" + std::to_string(idx) + "_Green.tiff.tif", 0);
		cv::calcOpticalFlowFarneback(prevImg, currImg, flow, 0.9, 1, 25, 2, 8, 1.2, 0);

		for (int i = 0; i < contour.size(); i++) {
			cv::Point2f flowXY = flow.at<cv::Point2f>(contour[i].y, contour[i].x);
			contour[i] = cv::Point(cvRound(contour[i].x + flowXY.x), cvRound(contour[i].y + flowXY.y));
		}

		cv::cvtColor(currImg, colourImg, cv::COLOR_GRAY2BGR);
		drawOptFlowMap(flow, colourImg, 16, cv::Scalar(0, 0, 255));
		cv::drawContours(colourImg, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 255));
		cv::imwrite(outBase + std::to_string(idx) + ".tiff", colourImg);

		prevImg = currImg.clone();
	}
}

static int getNthHighestIntensity(const cv::Mat& img) {
	//int asdf = img.type();
	std::unordered_set<int> us;
	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			us.insert(img.at<ushort>(r, c));
		}
	}
	std::vector<int> intensities(us.begin(), us.end());
	std::sort(intensities.begin(), intensities.end());
	//return intensities[intensities.size() - (size_t)(intensities.size() * 0.05)];
	return intensities[(size_t) (0.95 * intensities.size())];
}

static void modeSeeking() {
	int hsize = 65535;
	float hranges[] = {0, 65535};
	const float* phranges = hranges;

	prevImg = cv::imread(inBase + "14000_700_Green.tiff.tif", cv::IMREAD_UNCHANGED);
	ROI = cv::selectROI("Select ROI", prevImg);
	cv::destroyWindow("Select ROI");
	cv::Mat roiImg = prevImg(ROI);

	int maxval = getNthHighestIntensity(prevImg);
	cv::Mat mask = cv::Mat::zeros(prevImg.rows, prevImg.cols, prevImg.type());
	cv::threshold(roiImg, mask(ROI), maxval, 65535, cv::THRESH_BINARY);

	mask.convertTo(mask, CV_8U);
	cv::Mat hist;
	cv::calcHist(&roiImg, 1, 0, mask(ROI), hist, 1, &hsize, &phranges);
	//cv::normalize(hist, hist, 0, ROI.width * ROI.height, cv::NORM_MINMAX);

	cv::Mat backproj;
	cv::calcBackProject(&prevImg, 1, 0, hist, backproj, &phranges);
	//backproj &= mask;
	cv::normalize(backproj, backproj, 0, 65535, cv::NORM_MINMAX);

	cv::RotatedRect trackBox = cv::CamShift(backproj, ROI, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

	cv::imwrite(outBase + "700_backproj.tiff", backproj);
	cv::cvtColor(prevImg, colourImg, cv::COLOR_GRAY2BGR);
	cv::ellipse(colourImg, trackBox, cv::Scalar(0, 0, 65535), 2, cv::LINE_AA);
	cv::rectangle(colourImg, ROI, cv::Scalar(0, 65535, 0), 2, cv::LINE_4);
	cv::imwrite(outBase + "700_camshift.tiff", colourImg);

	for (int idx = 701; idx <= 910; idx++) {
		prevImg = cv::imread(inBase + std::to_string(idx * 20) + "_" + std::to_string(idx) + "_Green.tiff.tif", cv::IMREAD_UNCHANGED);

		cv::calcBackProject(&prevImg, 1, 0, hist, backproj, &phranges);
		cv::normalize(backproj, backproj, 0, 65535, cv::NORM_MINMAX);

		cv::RotatedRect trackBox = cv::CamShift(backproj, ROI, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

		cv::imwrite(outBase + std::to_string(idx) + "_backproj.tiff", backproj);
		cv::cvtColor(prevImg, colourImg, cv::COLOR_GRAY2BGR);
		cv::ellipse(colourImg, trackBox, cv::Scalar(0, 0, 65535), 2, cv::LINE_AA);
		cv::rectangle(colourImg, ROI, cv::Scalar(0, 65535, 0), 2, cv::LINE_4);
		cv::imwrite(outBase + std::to_string(idx) + "_camshift.tiff", colourImg);
	}
}

int main(void) {
	//opticalFlow();

	modeSeeking();

	return 0;
}