//
// Created by huiyan on 2021/2/11.
//

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

// nfeatures：期望提取的特征点个数
// nlevels：金字塔层数
// scaleFactor：相邻两层金字塔之间的相对尺度因子，大于1，金字塔越往上的图像每个像素代表的范围越大
// mvScaleFactor：累乘得到每一层相对第一层的尺度因子
// mvLevelSigma2：尺度因子mvScaleFactor的平方
// mvInvScaleFactor：尺度因子mvScaleFactor的逆
// mvInvLevelSigma2：尺度因子平方mvLevelSigma2的逆
// mnFeaturesPerLevel：记录每一层期望提取的特征点个数
// iniThFAST：提取fast特征点的默认阈值
// minThFAST：如果使用iniThFAST默认阈值提取不到特征点则使用最小阈值再次提取

namespace ORB_SLAM
{

}//namespace ORB_SLAM

#endif //ORBEXTRACTOR_H
