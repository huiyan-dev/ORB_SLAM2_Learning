//
// Created by huiyan on 2021/2/12.
//

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ORBextractor.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2
{
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    /**
     *
     * @param nfeatures 总共需要提取的特征数
     * @param scaleFactor   相邻金字塔层的尺度因子
     * @param nlevels   金字塔层数
     * @param iniThFAST 
     * @param minThFAST
     */
    ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
                               nfeatures(_nfeatures),scaleFactor(_scaleFactor),nLevels(_nlevels),
                               iniThFAST(_iniThFAST),minThFAST(_minThFAST)
    {
        //图像金字塔相关参数初始化
        mvScaleFactor.resize(_nlevels);
        mvLevelSigma2.resize(_nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for(int i = 1; i < _nlevels; ++i)
        {
            mvScaleFactor[i] = mvScaleFactor[i-1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }
        mvInvScaleFactor.resize(_nlevels);
        mvInvLevelSigma2.resize(_nlevels);
        for(int i = 0; i < _nlevels; ++i)
        {
            mvInvScaleFactor[i] = 1.0f / mvInvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvInvLevelSigma2[i];
        }
        mvImagePyramid.resize(_nlevels);
        mnFeaturesPerLevel.resize(_nlevels);

        //计算特征金字塔每层需要提取的特征点数

        //第i层提取的特征数 Ni , factor为第i层相对于第i+1层的尺度因子（scaleFactor是i+1层相对于i层的尺度因子）
        float factor = 1.0f / scaleFactor;

        // Level_0 的期望提取的特征点数
        //  第0层时i为8, 第1层时为7, ... , Ni = nfeatures * (1 - factor) / (1 - pow(factor, i))
        float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)_nlevels));

        // Level_1 - Level_7的期望提取的特征点数
        int nSumFeatures = 0;
        for (int level = 0; level < _nlevels - 1; ++level)
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            nSumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[_nlevels - 1] = std::max(nfeatures - nSumFeatures, 0);
        int count = 0;

        std::cout<<"#---------------------mnFeaturesPerLevel----------------------------------------------#"<<std::endl;
        for (int i = 0; i < _nlevels; ++i)
        {
            std::cout<<"mnFeaturesPerLevel["<<i<<"] = "<<mnFeaturesPerLevel[i]<<std::endl;
            count += mnFeaturesPerLevel[i];
        }
        std::cout<<"countSumFeatures = "<<count<<std::endl;
        std::cout<<"#---------------------mnFeaturesPerLevel----------------------------------------------#"<<std::endl;
    }
    void ORBextractor::ComputePyramid(cv::Mat image)
    {

        for(int level = 0; level < nLevels; ++level)
        {
            float scale = mvScaleFactor[level];
            Size sz(cvRound((float)image.cols * scale),
                    cvRound((float)image.rows * scale)
                    );
            Size wholeSize(sz.width + EDGE_THRESHOLD * 2,
                           sz.height + EDGE_THRESHOLD * 2
                    );
            Mat temp(wholeSize, image.type()), maskTemp;
            //初始图像, Level_0, 原始比例的图像
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            //其他层的图像
            if(level != 0)
            {
                resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
                copyMakeBorder(mvImagePyramid[level], temp,
                               EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED
                               );
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101
                        );
            }

        }
        std::cout<<"#---------------------ComputePyramid(image)----------------------------------------------#"<<std::endl;
        cv::imwrite("original.png", image);
        for(int i = 0; i < nLevels; ++i)
        {
            std::string fileName = "Level_" + std::to_string(i) + ".png";
            std::cout<<mvImagePyramid[i].size<<std::endl;
            cv::imwrite(fileName, mvImagePyramid[i]);
        }
        std::cout<<"#---------------------ComputePyramid(image)----------------------------------------------#"<<std::endl;
    }

    void ORBextractor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeyPoints)
    {
        std::cout<<allKeyPoints[1][1].pt<<std::endl;
    }
}



int main()
{
    ORB_SLAM2::ORBextractor orb_extractor = ORB_SLAM2::ORBextractor(1200,1.2,8,20,7);
    cv::Mat image = cv::imread("data/1.png");
    orb_extractor.ComputePyramid(image);
    vector<vector<KeyPoint>>    allKeyPoints;
    KeyPoint keyPoint;
    orb_extractor.ComputeKeyPointsOctTree(allKeyPoints);
    return 0;
}