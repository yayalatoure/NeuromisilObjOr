//
// Created by lalo on 30-08-18.
//

#include "foot.h"


foot::foot(bool start) {
    this -> start = true;
    Reset_R = false;
    Reset_L = false;
    step_R = false;
    step_L = false;
    occlusion = false;
}

//// Draw Foot Rectangles from Measurement ////
void foot::paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes){
    std::map<int, cv::Rect>::iterator it, it_end = bboxes.end();
    int i = 0;
    for(it = bboxes.begin(); it != it_end; it++) {
        i += 1;
        cv::rectangle(img, it->second, cv::Scalar(0,255,0), 2);
        if (i == 2)
            break;
    }

}

//// Get Bigger Blobs of Segmentation Image Lower Part ////
void foot::getBlobs(cv::Mat labels, std::map<int, cv::Rect> &bboxes) {

    int ro = labels.rows, co = labels.cols;
    int label, x, y;

    bboxes.clear();
    for(int j=0; j<ro; ++j)
        for(int i=0; i<co; ++i) {
            label = labels.at<int>(j,i);
            if(label > 0) {                    // Not Background?
                if(bboxes.count(label) == 0) { // New label
                    cv::Rect r(i,j,1,1);
                    bboxes[label] = r;
                } else {                       // Update rect
                    cv::Rect &r = bboxes[label];
                    x = r.x + r.width  - 1;
                    y = r.y + r.height - 1;
                    if(i < r.x)  r.x = i;
                    if(i > x)    x = i;
                    if(j < r.y)  r.y = j;
                    if(j > y)    y = j;
                    r.width  = x - r.x + 1;
                    r.height = y - r.y + 1;
                }
            }
        }
}

//// Get Foot Boxes from Blobs and Labels ////
void foot::getFeet(cv::Mat fg, std::map<int, cv::Rect> &bboxes, cv::Mat labels, cv::Mat labels2, std::map<int, cv::Rect> &fboxes){

    // Selecciona la región conectada más grande
    int Direc = 0, biggestblob = 1;
    string Direccion;

    getBlobs(labels, bboxes); // NOLINT

    for(unsigned int j=0; j < bboxes.size(); j++){
        if(bboxes[j].area() >= bboxes[biggestblob].area()) biggestblob = j;
    }

    // Crea una ROI en la parte inferior del jugador para visualizar sólo los
    // pies y eliminar el resto del análisis.

    Rect ROI;
    ROI.x = bboxes[biggestblob].x;
    ROI.y = int( bboxes[biggestblob].y + bboxes[biggestblob].height*0.8);
    ROI.height = int(bboxes[biggestblob].height*0.2);
    ROI.width = bboxes[biggestblob].width;

    Mat mask = Mat::zeros(fg.size(), CV_8U);
    rectangle(mask, ROI, Scalar(255), CV_FILLED);
    Mat fgROI = Mat::zeros(fg.size(), CV_8U);

    // copia fg a fgROI donde mask es distinto de cero.
    fg.copyTo(fgROI, mask);
    // aplica componentes conectados otra vez.
    cv::connectedComponents(fgROI, labels2, 8, CV_32S);
    getBlobs(labels2, fboxes);

}

//// Segmentation and Foot Boxes ////
void foot::findBoxes(){

    /* Inicializacion */
    Mat fg, labels, labels2, stats, centroids;

    double backgroundRatio = 0.7;
    double learningRate = 0.005; ////0.005
    double varThreshold = 80;
    int    nmixtures = 3;
    int    history = 200; ////150


    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);
    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);

    //// Start Segmentation ////
    mog->apply(frameAct.procesFrame, fg, 2*learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,6)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,5))); ////(4,6)
    cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);

    //if(start){
        getFeet(fg, frameAct.blobBoxes, labels, labels2, frameAct.footBoxes);
        found = frameAct.footBoxes[1].width > 0;
        frameAct.segmentedFrame  =  fg.clone();
    //}




}


//// Kalman Initialization////
void foot::KalmanInit(int pie){

    cv::KalmanFilter kf;
    if(pie == Right) {
        kf = kf_R;
    }else{
        kf = kf_L;
    }

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
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
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 1e-2f;// 5.0f
    kf.processNoiseCov.at<float>(21) = 1e-2f;// 5.0f
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));

}

//// Kalman Prediction ////
void foot::KalmanPredict(int pie, int dT){

    cv::KalmanFilter *kf;
    cv::Mat *state;
    cv::Rect *predRect;
    cv::Point *centerKalman;

    if(pie == Right) {
        kf = &kf_R;
        state = &state_R;
        predRect = &predRect_R;
        centerKalman = &centerKalman_R;
    }else{
        kf = &kf_L;
        state = &state_L;
        predRect = &predRect_L;
        centerKalman = &centerKalman_L;
    }

    /////// Prediction ///////
    *state = (*kf).predict();

    ////// Predicted Rect Red //////
    (*predRect).width = static_cast<int>((*state).at<float>(4));
    (*predRect).height = static_cast<int>((*state).at<float>(5));
    (*predRect).x = static_cast<int>((*state).at<float>(0) - (*state).at<float>(4)/2);
    (*predRect).y = static_cast<int>((*state).at<float>(1) - (*state).at<float>(5));

    //// Predicted Point ////
    (*centerKalman).x = static_cast<int>((*state).at<float>(0));
    (*centerKalman).y = static_cast<int>((*state).at<float>(1));

    (*kf).transitionMatrix.at<float>(2) = dT;
    (*kf).transitionMatrix.at<float>(9) = dT;

}

//// Measure Foot No Ocluded Case ////
void foot::measureFoot(int pie){
    if(occlusion){
        if(pie == Right) {
            centerMeasured_R.x = frameAct.footBoxes[Right].x + frameAct.footBoxes[Right].width / 4;
            centerMeasured_R.y = frameAct.footBoxes[Right].y + frameAct.footBoxes[Right].height;
        }else{
            centerMeasured_L.x = frameAct.footBoxes[Right].x + (frameAct.footBoxes[Right].width*3) / 4;
            centerMeasured_L.y = frameAct.footBoxes[Right].y + frameAct.footBoxes[Right].height;
        }
    }else{
        if(pie == Right) {
            centerMeasured_R.x = frameAct.footBoxes[Right].x + frameAct.footBoxes[Right].width / 2;
            centerMeasured_R.y = frameAct.footBoxes[Right].y + frameAct.footBoxes[Right].height;
        }else{
            centerMeasured_L.x = frameAct.footBoxes[Left].x + frameAct.footBoxes[Left].width / 2;
            centerMeasured_L.y = frameAct.footBoxes[Left].y + frameAct.footBoxes[Left].height;
        }
    }
}

//// Euclidean Distance ////
double foot::distance(cv::Point center_kalman, cv::Point center_measured) {
    double dx = 0, dy = 0, result = 0;
    dx = pow(center_kalman.x - center_measured.x, 2);
    dy = pow(center_kalman.y - center_measured.y, 2);
    result = sqrt(dx + dy);
    return result;
}

//// Error Measure ////
void foot::measureError1Np(int pie){
    if (pie == Right){
        errorNpAct1_R = distance(centerKalman_R, centerMeasured_R);
    }else{
        errorNpAct1_L = distance(centerKalman_L, centerMeasured_L);
    }
}

//// Reset Kalman ////
void foot::KalmanResetStep(int pie){

    double error;
    bool reset;

    if (pie == Right){
            Reset_R = abs(errorNpAct1_R) > 2;
            reset = Reset_R;
    }else{
            Reset_L = abs(errorNpAct1_L) > 2;
            reset = Reset_L;
    }

    if(!occlusion & !reset){
        if (pie == Right){
            error = (errorNpAct1_R); //+ errorNpAnt1_R)/2;
            if(abs(error) < 2.5) {
                step_R = true;
            }
            errorNpAnt1_R = errorNpAct1_R;
        }else{
            error = (errorNpAct1_L); //  + errorNpAnt1_L)/2;
            if(abs(error) < 2.5) {
                step_L = true;
            }
            errorNpAnt1_L = errorNpAct1_L;
        }

    }

}

//// Kalman Correction ////
void foot::KalmanUpdate(int pie){

    cv::KalmanFilter *kf;
    cv::Mat *state;
    cv::Mat *measure;
    bool *reset;

    if(pie == Right) {
        kf = &kf_R;
        state = &state_R;
        measure = &meas_R;
        reset = &Reset_R;
    }else{
        kf = &kf_L;
        state = &state_L;
        measure = &meas_L;
        reset = &Reset_L;
    }

    // Cuando no encuentra caja
    if (frameAct.footBoxes[1].width <= 0){
        notFoundCount++;
        if( notFoundCount >= 100 ){
            found = false;
        }else{
            (*kf).statePost.at<float>(0) = (*state).at<float>(0);
            (*kf).statePost.at<float>(1) = (*state).at<float>(1);//// + (*state).at<float>(5);
            (*kf).statePost.at<float>(2) = (*state).at<float>(2);
            (*kf).statePost.at<float>(3) = (*state).at<float>(3);
            (*kf).statePost.at<float>(4) = (*state).at<float>(4);
            (*kf).statePost.at<float>(5) = (*state).at<float>(5);
        }
    }else{

        if (occlusion) {
            if(pie == Right) {
                (*measure).at<float>(0) = frameAct.footBoxes[Right].x + float(frameAct.footBoxes[Right].width) / 4;
                (*measure).at<float>(1) = frameAct.footBoxes[Right].y + float(frameAct.footBoxes[Right].height);
                (*measure).at<float>(2) = (float) frameAct.footBoxes[Right].width / 3;
                (*measure).at<float>(3) = (float) (frameAct.footBoxes[Right].height*3) /4 ;
            }else if(pie == Left){
                (*measure).at<float>(0) = frameAct.footBoxes[Right].x + (float(frameAct.footBoxes[Right].width)*3) / 4;
                (*measure).at<float>(1) = frameAct.footBoxes[Right].y + (float(frameAct.footBoxes[Right].height));
                (*measure).at<float>(2) = (float) frameAct.footBoxes[Right].width / 3;
                (*measure).at<float>(3) = (float) (frameAct.footBoxes[Right].height*3) /4 ;
            }
        } else {
            if(pie == Right) {
                (*measure).at<float>(0) = frameAct.footBoxes[Right].x + float(frameAct.footBoxes[Right].width) / 2;
                (*measure).at<float>(1) = frameAct.footBoxes[Right].y + float(frameAct.footBoxes[Right].height);
                (*measure).at<float>(2) = (float) frameAct.footBoxes[Right].width;
                (*measure).at<float>(3) = (float) frameAct.footBoxes[Right].height;
            }else if(pie == Left){
                (*measure).at<float>(0) = frameAct.footBoxes[Left].x + float(frameAct.footBoxes[Left].width) / 2;
                (*measure).at<float>(1) = frameAct.footBoxes[Left].y + float(frameAct.footBoxes[Left].height);
                (*measure).at<float>(2) = (float) frameAct.footBoxes[Left].width;
                (*measure).at<float>(3) = (float) frameAct.footBoxes[Left].height;
            }
        }
        // cambie flag reset por found
        if (*reset){ // First detection!
            // >>>> Initialization
            (*kf).errorCovPre.at<float>(0) = 1; // px
            (*kf).errorCovPre.at<float>(7) = 1; // px
            (*kf).errorCovPre.at<float>(14) = 1;
            (*kf).errorCovPre.at<float>(21) = 1;
            (*kf).errorCovPre.at<float>(28) = 1; // px
            (*kf).errorCovPre.at<float>(35) = 1; // px

            (*state).at<float>(0) = (*measure).at<float>(0);
            (*state).at<float>(1) = (*measure).at<float>(1);
            (*state).at<float>(2) = 0;
            (*state).at<float>(3) = 0;
            (*state).at<float>(4) = (*measure).at<float>(2);
            (*state).at<float>(5) = (*measure).at<float>(3);
            // <<<< Initialization

            (*kf).statePost.at<float>(0) = (*state).at<float>(0);
            (*kf).statePost.at<float>(1) = (*state).at<float>(1);
            (*kf).statePost.at<float>(2) = (*state).at<float>(2);
            (*kf).statePost.at<float>(3) = (*state).at<float>(3);
            (*kf).statePost.at<float>(4) = (*state).at<float>(4);
            (*kf).statePost.at<float>(5) = (*state).at<float>(5);

            *reset = false;

        }else{
            (*kf).correct((*measure)); // Kalman Correction
        }
        notFoundCount = 0;
    }

}


//// Generate Template ////
void foot::generateTemplateNp(){

    int xr, yr, wr, hr;
    int xl, yl, wl, hl;
    int offsetR = 1, offsetL = 1;

    if (frameAct.footBoxes[Right].width > 0 && frameAct.footBoxes[Left].width > 0){

        xr = frameAct.footBoxes[Right].x - offsetR; yr = frameAnt.footBoxes[Right].y - offsetR;
        wr = frameAnt.footBoxes[Right].width + 2*offsetR; hr = frameAnt.footBoxes[Right].height + 2*offsetR;

        xl = frameAnt.footBoxes[Left].x - offsetL; yl = frameAnt.footBoxes[Left].y - offsetL;
        wl = frameAnt.footBoxes[Left].width + 2*offsetL; hl = frameAnt.footBoxes[Left].height + 2*offsetL;

        if (xr > 0 && xl > 0 && yr > 0 && yl > 0){

            Rect roifootR(xr, yr, wr, hr);
            Rect roifootL(xl, yl, wl, hl);

            frameAct.templateFrameR = frameAct.procesFrame(roifootR);
            frameAct.tempmaskFrameR = frameAct.segmentedFrame(roifootR);

            frameAct.templateFrameL = frameAct.procesFrame(roifootL);
            frameAct.tempmaskFrameL = frameAct.segmentedFrame(roifootL);

        }
    }

}

//// OCCLUSION ////

//// Matching Score Partial Occlusion
////gets the matching score for Right and Left foot.
void foot::matchingScorePocc(){

    int offset_oc = 10;
    int xoc = frameAct.footBoxes[1].x - offset_oc,  yoc = frameAct.footBoxes[1].y - offset_oc;
    int woc = frameAct.footBoxes[1].width + 2*offset_oc, hoc = frameAct.footBoxes[1].height + 2*offset_oc;

    cv::Rect roioc(xoc, yoc, woc, hoc);

    frameAct.occlusionFrame = frameAct.procesFrame(roioc);
    frameAct.occlumaskFrame = frameAct.segmentedFrame(roioc);

    cv::Mat matchScoreR, matchScoreL, matchScoreShowR, matchScoreShowL;

    matchTemplate(frameAct.occlusionFrame, frameAnt.templateFrameR, matchScoreR, CV_TM_SQDIFF_NORMED);
    matchTemplate(frameAct.occlusionFrame, frameAnt.templateFrameL, matchScoreL, CV_TM_SQDIFF_NORMED);

    matchScoreR = 1 - matchScoreR;
    matchScoreL = 1 - matchScoreL;
    normalize(matchScoreR, matchScoreR, 255, 0, NORM_MINMAX);
    normalize(matchScoreL, matchScoreL, 255, 0, NORM_MINMAX);
    matchScoreR.convertTo(matchScoreR, CV_8UC1);     // NOLINT
    matchScoreL.convertTo(matchScoreL, CV_8UC1);     // NOLINT
    matchScoreR.convertTo(matchScoreShowR, CV_8UC1); // NOLINT
    matchScoreL.convertTo(matchScoreShowL, CV_8UC1); // NOLINT
    applyColorMap(matchScoreShowR, matchScoreShowR, COLORMAP_JET);
    applyColorMap(matchScoreShowL, matchScoreShowL, COLORMAP_JET);

    frameAct.matchScoreR = matchScoreR;
    frameAct.matchScoreShowR = matchScoreShowR;
    frameAct.matchScoreL = matchScoreL;
    frameAct.matchScoreShowL = matchScoreShowL;

}




















//// Draw Results to Image ////
void foot::drawingResults(){

    //// Foots Rectangles ////
    paintRectangles(frameAct.resultFrame, frameAct.footBoxes);

    //// Measured Centers ////
    cv::circle(frameAct.resultFrame, centerMeasured_R, 2, CV_RGB(0,255,0), -1);
    cv::circle(frameAct.resultFrame, centerMeasured_L, 2, CV_RGB(0,255,0), -1);

    //// Kalman Prediction ////
    cv::rectangle(frameAct.resultFrame, predRect_R, CV_RGB(255,0,0), 2);
    cv::rectangle(frameAct.resultFrame, predRect_L, CV_RGB(255,0,0), 2);
    cv::circle(frameAct.resultFrame, centerKalman_R, 2, CV_RGB(255,0,0), -1);
    cv::circle(frameAct.resultFrame, centerKalman_L, 2, CV_RGB(255,0,0), -1);

    //// Step Normal Detection ////
    if (step_R && !occlusion){
        cv::rectangle(frameAct.resultFrame, frameAct.footBoxes[Right], CV_RGB(0, 0, 255), 2);
        cv::circle(frameAct.resultFrame, centerMeasured_R, 2, CV_RGB(0,0,255), -1);
    }
    if (step_L && !occlusion){
        cv::rectangle(frameAct.resultFrame, frameAct.footBoxes[Left], CV_RGB(0, 0, 255), 2);
        cv::circle(frameAct.resultFrame, centerMeasured_L, 2, CV_RGB(0,0,255), -1);
    }


    //// Matchscore Partial Occlusion ////
    if (occlusion){

        namedWindow("Occlusion", WINDOW_AUTOSIZE);
        namedWindow("Template R", WINDOW_AUTOSIZE);
        namedWindow("Template L", WINDOW_AUTOSIZE);
        namedWindow("Matchscore R", WINDOW_AUTOSIZE);
        namedWindow("Matchscore L", WINDOW_AUTOSIZE);

        Size sizeoccBox(frameAct.occlusionFrame.cols*10, frameAct.occlusionFrame.rows*10);
        Size sizetempBoxR(frameAnt.templateFrameR.cols*10, frameAnt.templateFrameR.rows*10);
        Size sizetempBoxL(frameAnt.templateFrameL.cols*10, frameAnt.templateFrameL.rows*10);
        Size sizematchScoreR(frameAct.matchScoreShowR.cols*10, frameAct.matchScoreShowR.rows*10);
        Size sizematchScoreL(frameAct.matchScoreShowL.cols*10, frameAct.matchScoreShowL.rows*10);

        resize(frameAct.occlusionFrame, frameAct.occlusionFrame, sizeoccBox);
        resize(frameAnt.templateFrameR, frameAnt.templateFrameR, sizetempBoxR);
        resize(frameAnt.templateFrameL, frameAnt.templateFrameL, sizetempBoxL);
        resize(frameAct.matchScoreShowR, frameAct.matchScoreShowR, sizematchScoreR);
        resize(frameAct.matchScoreShowL, frameAct.matchScoreShowL, sizematchScoreL);

        imshow("Occlusion", frameAct.occlusionFrame);
        imshow("Template R", frameAnt.templateFrameR);
        imshow("Template L", frameAnt.templateFrameL);
        imshow("Matchscore R", frameAct.matchScoreShowR);
        imshow("Matchscore L", frameAct.matchScoreShowL);

    }



}