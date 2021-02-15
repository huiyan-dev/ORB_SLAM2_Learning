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
        /**
         * 构造函数
         * @param nfeatures 总共需要提取的特征数
         * @param scaleFactor   相邻金字塔层的尺度因子
         * @param nlevels   金字塔层数
         * @param iniThFAST 关键点提取的默认阈值
         * @param minThFAST 默认阈值无法提取到关键点时候使用的最小阈值
         */
        ORBextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST);

        ~ORBextractor(){}

        /**
         * 计算特征金字塔
         * @param image 输入的图像
         */
        void ComputePyramid(cv::Mat image);

        /**
         * 计算所有层图像的特征点
         * @param allKeyPoints  保存所有关键点的变量
         */
        void ComputeKeyPointsOctTree(std::vector< std::vector<cv::KeyPoint> > &allKeyPoints);

        /**
         * 利用四叉树裁剪提取到的关键点, 避免出现某个区域提取的关键点过于密集
         * @param vToDistributeKeys 当前层的所有关键点
         * @param minX 图像起始坐标X
         * @param maxX 图像终止坐标X
         * @param minY 图像起始坐标Y
         * @param maxY 图像终止坐标Y
         * @param N 需要提取的关键点总数(用来预分配空间)
         * @param level 当前关键点所属的金字塔层级
         * @return 除后的特征点(返回的坐标是当前区域图像的坐标)
         */
        std::vector<cv::KeyPoint> DistributeOctTree(    //使用四叉树裁剪提取的所有ORB特征点
                const std::vector<cv::KeyPoint> &vToDistributeKeys,
                const int &minX, const int &maxX, const int &minY, const int &maxY,
                const int &N, const int &level
                );

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

        std::vector<cv::Mat> mvImagePyramid;    //包含边界的特征金字塔图像
    };  //class ORBextractor

    class ExtractorNode
    {
    public:
        ExtractorNode():bNoMore(false){}    //默认不需要细分

        //根据当前结点细分为4个子结点
        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;    //结点中包含的所有关键点
        cv::Point2i UL, UR, BL, BR; //结点区域的顶点
        std::list<ExtractorNode>::iterator lit; //记录当前
        bool bNoMore;
    };  //class ExtractorNode
}

#endif //ORBEXTRACTOR_H
