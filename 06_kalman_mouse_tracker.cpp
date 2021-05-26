/*
 * MIT Licence.
 * Written by Milind Deore <tomdeore@gmail.com>
 *
 * Smooth, fluctuating mouse movements. It can be used elsewhere as well. 
 *
 * Compile:
 * g++ -std=c++11 06_kalman_mouse_tracker.cpp `pkg-config --cflags --libs opencv` -o kalman
 *
 * Run:
 *  ./kalman
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>


typedef struct mouse_info
{
    int x;
    int y;
}mouse_info_t;

void draw_lines(cv::Mat img, std::vector<cv::Point> pts, int r, int g, int b)
{
    cv::polylines(img, pts, 0, cv::Scalar(r, g, b));
}

cv::Mat new_image()
{
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    return img; 
}

class EKF
{
	private:
		const float pval = 0.1;
	    const float qval = 0.0001;
	    const float rval = 0.1;
        uint32_t height;
        uint32_t width;

		// No previous prediction noise covariance
		cv::Mat P_pre;
		cv::Mat x;
		cv::Mat P_post;
		cv::Mat Q;
		cv::Mat R;
		cv::Mat I;

		cv::Mat F;
		cv::Mat h;
		cv::Mat H;
		cv::Mat G;

	public:

		//
		// Constructor 
		//
		EKF(uint32_t n, uint32_t m)
		{
            width = n;
            height = m;

			x = cv::Mat::zeros(1, n, CV_32FC1);
			P_post = cv::Mat::eye(n, n, CV_32FC1) * pval;

			Q = cv::Mat::eye(n, n, CV_32FC1) * qval;
			R = cv::Mat::eye(m, m, CV_32FC1) * rval;

			I = cv::Mat::eye(n, n, CV_32FC1);
		}

		// 
		// Step 
		//

		cv::Mat step(float xx, float yy)
		{
            float xxyy[] = {xx, yy};
            cv::Mat z(1, 2, CV_32FC1, xxyy);

			// Predict
			F = cv::Mat::eye(2, 2, CV_32FC1);
			P_pre = F * P_post * F.t() + Q;

			// Update
			x.copyTo(h);
			H = cv::Mat::eye(2, 2, CV_32FC1);	

			cv::Mat H_P_pre_R =  H * P_pre * H.t() + R; 
			G =  P_pre * H.t() * H_P_pre_R.inv(); 
			x += (z - h) * G;
			P_post = (I - (G * H)) * P_pre;

			return x;
		}
};


//
// Globals
//
std::vector<cv::Point> measured_pts;
std::vector<cv::Point> kalman_pts;
EKF ekf(2, 2);

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    mouse_info_t* m_info = (mouse_info_t*)userdata;

    if ( event == cv::EVENT_MOUSEMOVE ) {
        m_info->x = x;
        m_info->y = y;
        std::cout << "Mouse move over the window - position (" << x << ", " << y   << ")" << std::endl;

    }
    else {
        std::cout << "Unsupported event" << std::endl;
    }

    cv::Mat img = new_image();

    // Grab current mouse position and add it to the trajectory
    cv::Point point = cv::Point(m_info->x, m_info->y);
    measured_pts.push_back(point);

    cv::Mat est_pts = ekf.step((float)m_info->x, (float)m_info->y);

    cv::Point kl_point = cv::Point(est_pts.at<float>(0,0), est_pts.at<float>(0,1));
    kalman_pts.push_back(kl_point);

    draw_lines(img, kalman_pts, 0, 255, 0);
    draw_lines(img, measured_pts, 255, 255, 0);

    imshow("Kalman Mousetracker [ESC to quit]", img);
    if (cv::waitKey(0) == 27) {
        exit(0);
    }

}



int main()
{
    cv::Mat img = new_image();
    cv::namedWindow("Kalman Mousetracker [ESC to quit]", 1);

    mouse_info_t m_info = {-1, -1};

    // 
    // Set the callback function for any mouse event, we are interested 
    // in movements only.
    cv::setMouseCallback("Kalman Mousetracker [ESC to quit]", 
                         CallBackFunc, &m_info);

    while (1)
    {
        imshow("Kalman Mousetracker [ESC to quit]", img);
        if (cv::waitKey(0) == 27) {
            std::cout << "Got 'ESC' singal" << std::endl;
            exit(0);
        }
    }

	return 0;
}
