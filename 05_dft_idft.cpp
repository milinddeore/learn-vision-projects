#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

/*
 * Compile: 
 * g++ -std=c++11 05_dft_idft.cpp -L/usr/local/lib -lopencv_highgui -lopencv_imgcodecs 
 * -lopencv_imgproc -lopencv_core -I/usr/local/include/opencv -I/usr/local/include 
 * -o dft_idft
 *
 *  Run:
 *  ./dft_idft
 */  

cv::Mat 
lgt_Dft(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat expand;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);

    cv::copyMakeBorder(src, expand, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT);

    cv::Mat planes[] = { cv::Mat_<float>(expand), cv::Mat::zeros(expand.size(), CV_32F) };

    /* Add to the expanded another plane with zeros */
    cv::merge(planes, 2, dst);

    /* Result may fit in the source matrix */
    cv::dft(dst, dst);

    /* Compute the magnitude and switch to logarithmic scale
     * => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
     * planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
     */
    cv::split(dst, planes);                   
    
    /* Planes[0] = magnitude */
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    
    /* Switch to logarithmic scale */
    mag += cv::Scalar::all(1);
    log(mag, mag);

    /* Crop the spectrum, if it has an odd number of rows or columns */
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));

    /* Rearrange the quadrants of Fourier image  so that the origin is at the 
     * image center 
     */
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    /* Swap quadrants (Top-Left with Bottom-Right and Top-Right with Bottom-Left) */
    cv::Mat tmp;                           
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    /* Transform the matrix with float values into a viewable image 
     * form (float between values 0 and 1).
     */ 
    cv::normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;
}


int main(int argc, char** argv) {
    cv::Mat img = cv::imread("./images/Snap.JPG");

    if (img.empty())
    {
        return -1;
    }

    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::imshow("Image", img);

    /* Calculate the dft */
    cv::Mat complex_img, mag;
    mag = lgt_Dft(img, complex_img);
    cv::imshow("Image DFT", mag);

    /* Calculating the idft (inverse) */
    cv::Mat inverseTransform;
    cv::dft(complex_img, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    cv::imshow("Reconstructed", inverseTransform);

    cv::waitKey();
    return 0;
}
