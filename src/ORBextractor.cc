//
// Created by huiyan on 2021/2/12.
//

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

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

    std::cout<<"#---------------------start : mnFeaturesPerLevel----------------------------------------------#"<<std::endl;
    for (int i = 0; i < _nlevels; ++i)
    {
        std::cout<<"mnFeaturesPerLevel["<<i<<"] = "<<mnFeaturesPerLevel[i]<<std::endl;
        count += mnFeaturesPerLevel[i];
    }
    std::cout<<"countSumFeatures = "<<count<<std::endl;
    std::cout<<"#---------------------end : mnFeaturesPerLevel----------------------------------------------#"<<std::endl;
}
void ORBextractor::ComputePyramid(cv::Mat image)
{
    std::cout<<"#---------------------start : ComputePyramid(image)----------------------------------------------#"<<std::endl;
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

    for(int i = 0; i < nLevels; ++i)
    {
        //std::string fileName = "Level_" + std::to_string(i) + ".png";
        std::cout<<mvImagePyramid[i].size<<std::endl;
        //cv::imwrite(fileName, mvImagePyramid[i]);
    }
    std::cout<<"#---------------------end : ComputePyramid(image)----------------------------------------------#"<<std::endl;

}

void ORBextractor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeyPoints)
{
    std::cout<<"#---------------------start : ComputeKeyPointsOctTree(allKeyPoints)----------------------------------------------#"<<std::endl;
    allKeyPoints.resize(nLevels);

    const float W = 30; //分区域的大小,30像素

    for(int level = 0; level < nLevels; ++level)    //计算所有图层的特征点
    {
        const int minBorderX = EDGE_THRESHOLD - 3;  //FAST角点需要半径为3的圆
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - (EDGE_THRESHOLD - 3);
        const int maxBorderY = mvImagePyramid[level].rows - (EDGE_THRESHOLD - 3);
        std::cout<<"(minBorderX, maxBorderX, minBorderY, maxBorderY) : ("
            <<minBorderX<<", "<<maxBorderX<<", "<<minBorderY<<", "<<maxBorderY<<")"<<std::endl;
        vector<cv::KeyPoint> vToDistributeKeys; //需要分配的所有关键点
        vToDistributeKeys.resize(nfeatures * 10);

        const float width = maxBorderX - minBorderX;    //提取特征的区域
        const float height = maxBorderY - minBorderY;

        //划分区域, 不足区域大小 W 的调整区域大小为 (wCell×hCell), 一共有 (nCols*nRows) 个区域
        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / W);
        const int hCell = ceil(height / W);

        //分区域提取特征点, 存储到 vToDistributeKeys
        for(int row = 0; row < nRows; ++row)
        {
            const float iniY = minBorderY + row * hCell;    //Y方向初始坐标 iniY
            float maxY = iniY + hCell + 6;    //Y方向终止坐标 maxY, 6 是上下两个边界的FAST角点半径为3的圆

            //如果初始坐标不在有效区域内(不包括提取FAST角点需要的圆), continue
            if(iniY > maxBorderY - 3) continue;
            //如果Y方向最大坐标maxY 超出了最大边界, maxY = maxBorderY
            if(maxY > maxBorderY)   maxY = maxBorderY;

            for(int col = 0; col < nCols; ++col)
            {
                //和计算 Y 方向类似
                const float iniX = minBorderX + col * wCell;
                float maxX = iniX + wCell + 6;
                if(iniX > maxBorderX - 3)   continue;   //这里 orb_slam2 源码上是 -6 , -3才应该是正确的
                if(maxX > maxBorderX)   maxX = maxBorderX;

                // opencv的features2d模块
                //在当前区域(iniX,iniY) - (maxX,maxY)提取关键点, 开启非极大值抑制
                vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                     vKeysCell, iniThFAST, true);

                //如果使用默认阈值 iniThFAST 提取不到关键点, 则使用最小阈值提取特征点
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, minThFAST, true);
                }

                //将提取到的关键点加入 vToDistributeKeys
                if(!vKeysCell.empty())
                {
                    for(vector<KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); ++vit)
                    {
                        (*vit).pt.x += col * wCell; //提取的 pt 是局部区域的坐标, 根据区域计算关键点在原图像的位置
                        (*vit).pt.y += row * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }   //分区域提取特征点


        //根据 mnFeaturesPerLevel(当前层需要提取的特征数) ,对提取的特征进行剔除
        vector<cv::KeyPoint> & keyPoints = allKeyPoints[level];
        keyPoints.reserve(nfeatures);   //预分配足够大的长度 nfeaetures
        // **************************************************************************
        std::cout<< "level_"<<level<<" start : DistributeOctTree(...)"<< std::endl;
        keyPoints = DistributeOctTree(  //得到剔除后的特征点(返回的坐标是当前区域图像的坐标) ************
                vToDistributeKeys,
                minBorderX, maxBorderX, minBorderY, maxBorderY,
                mnFeaturesPerLevel[level], level
                );
        std::cout<< "level_"<<level<<" end : DistributeOctTree(...)"<< std::endl;
        // **************************************************************************
        //恢复关键点在原图像的坐标
        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        const int nkps = keyPoints.size();
        for(int i = 0; i < nkps; ++i)
        {
            keyPoints[i].pt.x+=minBorderX;  //加上起始坐标
            keyPoints[i].pt.y+=minBorderY;
            keyPoints[i].octave=level;  //关键点所在金字塔的层数
            keyPoints[i].size = scaledPatchSize;    //缩放的面片大小, 与计算方向有关
        }

    }   //计算所有图层的特征点
    int sum_features_extract = 0;
    for(int i = 0; i < nLevels; i++)
    {
        sum_features_extract += allKeyPoints[i].size();
    }
    std::cout<<sum_features_extract<<"features have extracted"<<std::endl;
    std::cout<<"#---------------------end : ComputeKeyPointsOctTree(allKeyPoints)----------------------------------------------#"<<std::endl;
}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    //-----------------1 计算初始条件-----------------------
    //当前层图像的宽高比(必须大于0.5, 不然hX为0)取整得到初始结点的数目
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));
    //初始结点 X 的大小(单位为像素)
    const int hX = static_cast<float>(maxX - minX) / nIni;

    //lNodes存放结点数据, 只保留叶子结点, 一个结点是一个小区域, UL、UR、BL、BR是四个顶点坐标,vKeys保存该区域所有的关键点
    list<ExtractorNode> lNodes;
    //初始结点, x/hX 能根据x的值快速对应所属的结点
    vector<ExtractorNode *> vpIniNodes;
    vpIniNodes.resize(nIni);

    //-----------------2 初始化初始结点 : 结点区域, 预分配空间-----------------------
    for (int i = 0; i < nIni; ++i) {
        ExtractorNode temp;
        temp.UL = cv::Point2i(hX * static_cast<float>(i), 0);   //左上角顶点
        temp.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);   //右上角顶点
        temp.BL = cv::Point2i(temp.UL.x, maxY - minY);  //左下角顶点
        temp.BR = cv::Point2i(temp.UR.x, temp.BR.y);    //右下角顶点
        temp.vKeys.reserve(vToDistributeKeys.size());   //预留该结点内所有关键点的容器大小

        lNodes.push_back(temp); //当前结点加入lNodes
        vpIniNodes[i] = &lNodes.back(); //初始结点对应存放当前结点的引用, 后续可以通过 X 坐标快速判断属于哪一个结点
    }
    //-----------------2 初始化初始结点 : 关联所有结点内的关键点-----------------------
    for (size_t i = 0; i < vToDistributeKeys.size(); ++i) {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)    //如果结点内只有一个关键点, 不需要细分结点
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty()) //如果结点内关键点为空, 删除该结点
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    //-----------------3 使用四叉树方法对关键点进行裁剪 ： 初始化相关变量-----------------------
    bool bFinish = false;
    int epoch = 0;

    vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode; //用来保存细分结点的信息, 四叉树有四个子结点
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while(!bFinish)
    {
        ++epoch;
        int prevSize = lNodes.size();   //当前的结点数
        lit = lNodes.begin();
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();  //vector.size 属性置为 0
        // -----------------3 使用四叉树方法对关键点进行裁剪 ： 遍历所有结点, 细分满足条件的结点-----------------------
        while(lit != lNodes.end())
        {
            if(lit->bNoMore)    //如果当前结点不需要细分, 跳出当前结点
            {
                lit++;
                continue;
            }
            else    //否则,继续细分成四个子区域
            {
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                if(n1.vKeys.size() > 0)   // 如果结点中有关键点则加入 lNodes
                {
                    lNodes.push_front(n1);  //后来后出,插在前面
                    if(n1.vKeys.size() > 1)   //如果结点中关键点数目大于 1 , 则使用 nToExpand计数, 后续并继续细分该结点
                    {
                        nToExpand++;
                        // pair<节点中关键点个数, 节点索引> , 后续可以直接通过关键点的 x 坐标快速计算到其属于哪个结点
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        // 记录节点自己的迭代器指针
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                // 该节点已经分裂完，删除该节点
                lit=lNodes.erase(lit);
                continue;
            }
        }

        // -----------------4 终止条件 ： 已经得到需要提取的关键点数量-----------------------
        if((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        // -----------------4 终止条件 ： 未达到需要提取的关键点数量, 继续细分-----------------------
        else if(((int)lNodes.size() + nToExpand * 3) > N)     // 当再划分之后所有的Node数快接近要求数目时，优先对包含特征点比较多的区域进行划分
        {
            while(!bFinish)
            {
                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                // 对需要划分的部分进行排序, 即对兴趣点数较多的区域进行划分
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());

                for(int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    if(n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;

            }
        }
    }

    // -----------------5 现在一共得到来 nFeatures 个区域, 保留区域内最好的一个关键点(pKP.response最大的点)-----------------------
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(N);
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); ++lit) {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint *pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); ++k) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        vResultKeys.push_back(*pKP);
    }
    std::cout<<"level_"<<level<<" nodes : "<<lNodes.size()<<std::endl;
    return vResultKeys;
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
}// ExtractorNode::DivideNode

}   //namespace ORB_SLAM2




int main()
{
ORB_SLAM2::ORBextractor orb_extractor = ORB_SLAM2::ORBextractor(1200,1.2,8,20,7);
cv::Mat image = cv::imread("data/1.png");
orb_extractor.ComputePyramid(image);
vector<vector<KeyPoint>>    allKeyPoints;
orb_extractor.ComputeKeyPointsOctTree(allKeyPoints);
std::vector<cv::Mat> outImg;
outImg.resize(8);
for(int i = 0; i < 8; ++i)
{
    cv::drawKeypoints(orb_extractor.mvImagePyramid[i], allKeyPoints[i], outImg[i], cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
}
for(int i = 0; i < 8; ++i)
{
    std::string fileName = "Level_" + to_string(i)+"_features.png";
    cv::imwrite(fileName, outImg[i]);
}
return 0;
}