//
// Created by huiyan on 2021/2/12.
//

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace ORB_SLAM2
{
    class ORBextractor
    {
    public:
        ORBextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST);
        ~ORBextractor(){}

        std::vector<cv::Mat> mvImagePyramid;

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector< std::vector<cv::KeyPoint> > &allKeyPoints);

        int nfeatures;  //需要提取的特征点数
        int nLevels;    //金字塔层数
        double scaleFactor; //相邻金字塔层的尺度因子
        int iniThFAST;  //提取fast特征点的默认阈值
        int minThFAST;  //如果默认阈值无法提取到预期的特征，则使用最小阈值minThFAST再次提取

        std::vector<int> mnFeaturesPerLevel;    //记录所有层金字塔期望提取的特征点个数

        std::vector<float> mvScaleFactor;   //每一层相对第一层的尺度因子
        std::vector<float> mvInvScaleFactor;    //mvScaleFactor的逆

        std::vector<float> mvLevelSigma2;   //尺度因子mvScaleFactor的平方
        std::vector<float> mvInvLevelSigma2; //mvLevelSigma2的逆

    };
}

#endif //ORBEXTRACTOR_H
