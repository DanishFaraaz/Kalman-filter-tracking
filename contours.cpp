#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "withrobot_camera.hpp"

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

RNG rng(12345);

int main()
{
    // Declaration of varaiables
    int count = 0, ind = 0;
    int max_area, cnt_id;
    Rect2d bbox;

    // Camera capture
    VideoCapture cap("polwhite0530.mp4");

    // Kalman filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    Mat state(stateSize, 1, type);
    Mat meas(measSize, 1, type);

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1;
    kf.processNoiseCov.at<float>(7) = 1;
    kf.processNoiseCov.at<float>(14) = 1;
    kf.processNoiseCov.at<float>(21) = 1;
    kf.processNoiseCov.at<float>(28) = 1;
    kf.processNoiseCov.at<float>(35) = 1;
    
    setIdentity(kf.measurementNoiseCov, Scalar(1e-4));

    // const char* devPath = "/dev/video2";
    // Withrobot::Camera camera(devPath);    // /* bayer RBG 640 x 480 80 fps */
    // camera.set_format(640, 480, Withrobot::fourcc_to_pixformat('G','B','G','R'), 1, 80);    // /*
    //  * get current camera format (image size and frame rate)
    //  */
    // Withrobot::camera_format camFormat;
    // camera.get_current_format(camFormat);    // /*
    //  * Print infomations
    //  */
    // std::string camName = camera.get_dev_name();
    // std::string camSerialNumber = camera.get_serial_number();    // printf("dev: %s, serial number: %s\n", camName.c_str(), camSerialNumber.c_str());
    // printf("----------------- Current format informations -----------------\n");
    // camFormat.print();
    // printf("---------------------------------------------------------------\n");    // int brightness = camera.get_control("Gain");
    // int exposure = camera.get_control("Exposure (Absolute)");    // //For indoors - gain=66, exposure=170
    // camera.set_control("Gain", 120);
    // //camera.set_control("Gain", brightness);
    // camera.set_control("Exposure (Absolute)", 100);
    // //camera.set_control("Exposure (Absolute)", exposure);    // std::string windowName = camName + " " + camSerialNumber;
    // cv::Mat srcImg(cv::Size(camFormat.width, camFormat.height), CV_8UC1);
    // cv::Mat frame(cv::Size(camFormat.width, camFormat.height), CV_8UC3);	Mat frame, dst, cdst, cdstP, abs_dst;

	int morph_elem = 0;
	int morph_size = 5;
	int morph_operator = 0;
	int operation = morph_operator + 2;

    double ticks = 0;
    bool found = false;   
    
    while (1)
    {

        // Get FPS
        double precTicks = ticks;
        ticks = (double)getTickCount();

        double dT = (ticks - precTicks) / getTickFrequency();

        Mat frame;
        cap >> frame;

        Mat copy;
        copy = frame.clone();

        cout << "Start of loop" << endl;

        
        // int size = camera.get_frame(srcImg.data, camFormat.image_size, 1);
        // if (size == -1) {
        //     printf("error number: %d\n", errno);
        //     perror("Cannot get image from camera");
        //     camera.stop();
        //     camera.start();
        //     continue;
        // }

        if (found) {
            
            // Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;

            // cout << "dT: " << endl << dT << endl;

            // state = kf.predict();
            // cout << "State post: " << endl << state << endl;

            Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            rectangle(frame, predRect, Scalar(255,0,0), 2, 8, 0);
            cout << "Kalman width: " << predRect.width << ", " << "Kalman height: " << predRect.height << endl;
            cout << "----------------------------------" << endl;
        }

        Mat gray, dst, cdst, cdsP;
        cvtColor(copy, gray, COLOR_BGR2GRAY);        
        threshold( gray, gray, 180, 255, 0 ); 		
        Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
 		morphologyEx( gray, gray, operation, element );

        vector<vector<Point>> contours, closed_cnts;
        vector<Vec4i> hierarchy;
        findContours(gray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

        vector<Rect> boundRect(contours.size());
        Scalar color = Scalar(0, 255, 0);

        // Displaying the maximum bounding box
        if (contours.size() > 1) {
            int area_func[contours.size()];
            for (int i=0; i<contours.size(); i++)
            {
                if (contourArea(contours[i])>50 && arcLength(contours[i], true) > 10)
                {
                    area_func[ind] = contourArea(contours[i]);
                }
            }
            max_area = *max_element(area_func, area_func+boundRect.size());
            cnt_id = std::distance(area_func,std::max_element(area_func, area_func + sizeof(area_func)/sizeof(area_func[0])));

            bbox = boundingRect(contours[cnt_id]);

            rectangle(frame,bbox, color, 2, 8, 0);
            count = count + 1;
        }
        else if (contours.size() == 1){
            for (int i=0; i<contours.size(); i++)
            {
                if (contourArea(contours[0])>50 && arcLength(contours[0], true) > 10)
                {
                    bbox = boundingRect(contours[0]);
                    rectangle(frame, bbox, color, 2, 8, 0);
                    count = count + 1;
                }
            }
        }
        else {
            count = 0;
            found = false;
        }

        //cout << bbox.area() << endl;

        // blur(gray, gray, Size(5,5));
		// Canny(gray, dst, 200, 220, 3);

        if (count > 50) {

            meas.at<float>(0) = bbox.x + bbox.width / 2;
            meas.at<float>(1) = bbox.y + bbox.height / 2;
            meas.at<float>(2) = (float)bbox.width;
            meas.at<float>(3) = (float)bbox.height;

            cout << "Detection width: " << bbox.width << "Detection height: " << bbox.height << endl;

            if (!found) {
                
                kf.errorCovPre.at<float>(0) = 1;
                kf.errorCovPre.at<float>(7) = 1;
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1;
                kf.errorCovPre.at<float>(35) = 1;

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(1) = 0;
                state.at<float>(1) = 0;
                state.at<float>(1) = meas.at<float>(2);
                state.at<float>(1) = meas.at<float>(3);

                kf.statePost = state;

                found = true;

            }
            else{
                kf.correct(meas);
            }

        }

        imshow("Operation", gray);
        imshow("Detection and Tracking", frame);
        waitKey(5);
    }
        
}