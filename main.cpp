

#include "foot.h"
#include <QApplication>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv/cv.h>
#include <QRect>
#include <QDir>

using namespace std;
using namespace cv;


int main(int argc, char *argv[]){

    cv::Mat img_cal, img_test, img_proc, img_show, seg_rect;

    //// Images Reading
//    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/CALIBRACION01/*.jpg";
//    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/TEST01/*.jpg";
//    std::string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/2_CAMARAS/FEED1/*.jpg";
//    std::string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/2_CAMARAS/FEED1/*.jpg";

    //// Videos nuevos
    string path_cal  = "/home/lalo/Desktop/Data_Videos/CAL_Test1/*.jpg";
    string path_test = "/home/lalo/Desktop/Data_Videos/Player3/*.jpg";


    //// Video Original
//    int count_test = 195+145, count_cal = 0, limit = 150-145;

    //// Videos nuevos limites nuevos
    int count_test = 2649-2335-100, count_cal = 0, limit = 100;

    vector<String> filenames_cal, filenames_test;

    glob(path_test, filenames_test);
    glob(path_cal , filenames_cal);


    int digits = 5;
    size_t pos = filenames_test[count_test].find(".jpg");

    //// Logging error Kalman (frames) ////
    string fileName, substring;
//    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil_Lalo/NeuroMisil/Logging/pasos_result.csv";
//    ofstream ofStream(fileName);
//    ofStream << "Frame" << "," << "CX_Paso" << "," << "CY_Paso" << "," << "W_Paso" << "." << "H_Paso" << "," << "Pie" << "\n";

    char ch = 0;
    int dT = 1;


    foot Foot(false);

    Foot.kalmanInit(Foot.Right);
    Foot.kalmanInit(Foot.Left);

    Foot.errorNpAct1_R = 0;



    while(ch != 'q' && ch != 'Q') {

        //// Transfer Frame Structure ////
        Foot.frameAnt = Foot.frameAct;

        ////////// Frame Acquisition /////////
        if (count_cal < limit) {
            img_cal = imread(filenames_cal[count_cal], CV_LOAD_IMAGE_COLOR);
            img_test = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos - digits);
            Foot.frameAct.processFrame = img_cal;
            Foot.start = true;

            cout << substring << "\n" << endl;

        } else {
            img_test = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos - digits);
            Foot.frameAct.processFrame = img_test;
            Foot.start = true;

            cout << substring << "\n" << endl;

        }

        ///// Algoritmo /////

        if (Foot.frameAct.processFrame.data) {

            //// Low Step Flag ////
            Foot.step_R = false;
            Foot.step_L = false;

            //////// 2D Feet Boxes ////////
            Foot.frameAct.resultFrame = Foot.frameAct.processFrame.clone();
            Foot.segmentation();


//            Foot.occlusion = bool(Foot.frameAct.footBoxes.size() <= 1);
//
//
//            if (Foot.found){
//
//                if (!Foot.occlusion){
//                    //// Measure Foot ////
//                    Foot.measureFoot(Foot.Right);
//                    Foot.measureFoot(Foot.Left);
//
//                    //// Kalman Filter ////
//                    Foot.kalmanPredict(Foot.Right, dT);
//                    Foot.kalmanPredict(Foot.Left, dT);
//
//                    //// Kalman Update ////
//                    Foot.kalmanUpdate(Foot.Right);
//                    Foot.kalmanUpdate(Foot.Left);
//
//                    //// Measure Error ////
//                    Foot.measureError1Np(Foot.Right);
//                    Foot.measureError1Np(Foot.Left);
//
//                    //// Kalman Reset Step ////
//                    Foot.kalmanResetStep(Foot.Right);
//                    Foot.kalmanResetStep(Foot.Left);
//
//                    //// Generate Template ////
//                    Foot.generateTemplateNp();
//
//                    //// Drawing Results ////
//                    Foot.drawingResults();
//
//
//
//                }else{
//                    //// matchingScorePocc ////
//                    Foot.matchingScorePocc();
//
//                    ////  One matchingAraea? ////
//                    Foot.occlusionType();
//                    //// Total Occlusion? ////
//                    //if (Foot.totalOccR && Foot.totalOccL){
//
//                        cout << "Total Occlusion" << endl;
//
//                    //}else{
//
//
//                    //// Kalman Filter ////
//                    Foot.kalmanPredict(Foot.Right, dT);
//                    Foot.kalmanPredict(Foot.Left, dT);
//
//                    //// Maximum Candidates Vector ////
//                    Foot.maxCandidatesPocc();
//
//                    //// Select Matching Score ////
//                    Foot.matchingSelectPocc();
//
//                    //// Proyect Predicted Boxes ////
//                    Foot.proyectBoxes();
//
//                    //// Measure Foot ////
//                    Foot.measureFoot(Foot.Right);
//                    Foot.measureFoot(Foot.Left);
//
//                    //// Kalman Update ////
//                    Foot.kalmanUpdate(Foot.Right);
//                    Foot.kalmanUpdate(Foot.Left);
//
//                    //// Measure Error ////
//                    Foot.measureError1Np(Foot.Right);
//                    Foot.measureError1Np(Foot.Left);
//
//                    //// Kalman Reset Step ////
//                    Foot.kalmanResetStep(Foot.Right);
//                    Foot.kalmanResetStep(Foot.Left);
//
//                    //// Generate Template ////
//                    Foot.generateTemplateNp();
//
//
//                    //}
//
//
//                    //// Drawing Results ////
//                    Foot.drawingResults();
//
//                    //// Clear Variables ////
//                    Foot.clearVariables();
//
//
//                }
//
//            }





        }

        /////// Visualize ///////
//        if (count_cal < limit)
//            if (img_test.data) cv::imshow("frameAct", Foot.frameAct.resultFrame);

        if (Foot.start && (Foot.frameAct.resultFrame.data) && (Foot.frameAnt.resultFrame.data)) {

            cv::imshow("frameAct",  Foot.frameAct.resultFrame);
            cv::imshow("frameAnt ", Foot.frameAnt.resultFrame);

            //// ver segmentacion con videos nuevos
            cv::imshow("frameSegAct ", Foot.frameAct.segmentedFrame);






        }

        count_cal++;
        count_test++;
        ch = char(cv::waitKey(0));



    }

    return 0;

}