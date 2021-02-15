vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                                     const int &minX, const int &maxX, const int &minY, const int &maxY,
                                                     const int &N, const int &level)
{
    std::cout<< "#---------------------start : DistributeOctTree(...)--------------------#"<< std::endl;

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
    list<ExtractorNode>::iterator lit = lNodes.begin();
    for (size_t i = 0; i < vToDistributeKeys.size(); ++i) {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }
    //-----------------2 初始化初始结点 : 筛选得到的 vpIniNodes-----------------------
    for (lit = lNodes.begin(); lit != lNodes.end(); ++lit) {
        if (1 == lit->vKeys.size()) lit->bNoMore = true;    //如果结点内只有一个关键点, 不需要细分结点
        if (lit->vKeys.empty()) lit = lNodes.erase(lit);    //如果结点内关键点为空, 删除该结点
    }
    //-----------------3 使用四叉树方法对关键点进行裁剪 ： 初始化相关变量-----------------------
    bool bFished = false;
    int epoch = 0;

    vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode; //用来保存细分结点的信息, 四叉树有四个子结点
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFished)    //四叉树细分开始
    {
        ++epoch;
        int prevSize = lNodes.size();   //当前的结点数
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();  //vector.size 属性置为 0
        // -----------------3 使用四叉树方法对关键点进行裁剪 ： 广度优先遍历所有结点, 细分满足条件的结点-----------------------
        lit = lNodes.begin();
        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                // 如果这个区域不止一个特征点，则进一步细分成四个子区域
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                // 如果子节点中包含特征点，则将该节点添加到节点链表中
                if(n1.vKeys.size()>0)
                {
                    // note：将新分裂出的节点插入到容器前面，迭代器后面的都是上一次分裂还未访问的节点
                    lNodes.push_front(n1);
                    // 如果该节点中包含的特征点超过1，则该节点将会继续扩展子节点，使用nToExpand统计接下来要扩展的节点数
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        // 按照 pair<节点中特征点个数，节点索引> 建立索引，后续通过排序快速筛选出包含特征点个数比较多的节点
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        // 记录节点自己的迭代器指针
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
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
        if ((int) lNodes.size() >= nFeatures || (int) lNodes.size() == prevSize) bFished = true;
        // -----------------4 终止条件 ： 未达到需要提取的关键点数量, 继续细分-----------------------
        else if (((int) lNodes.size() + nToExpand * 3) > nFeatures)    //再划分时优先划分关键点数量更多的结点
        {
            while(!bFished)
            {
                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                // 对需要划分的部分进行排序, 即对兴趣点数较多的区域进行划分
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());

                for(int j = vPrevSizeAndPointerToNode.size() - 1; j>=0; j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size() >= nFeatures)
                        break;
                }

                if((int)lNodes.size() >= nFeatures || (int)lNodes.size()==prevSize)
                    bFished = true;

            }
        }
    }   //四叉树细分结束

    // -----------------5 现在一共得到来 nFeatures 个区域, 保留区域内最好的一个关键点(pKP.response最大的点)-----------------------
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nFeatures);
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
    std::cout<< "#------------"<<lNodes.size()<<"---------end : DistributeOctTree(...)--------------------#"<< std::endl;
    return vResultKeys;
}
