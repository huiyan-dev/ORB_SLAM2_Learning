
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                     const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes
    // 图像大小一般为矩形，且宽高比不是整数，
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    // note：如果图像的宽不到高的一半，hX=0，会出问题，据此推断，这里默认为图像宽大于高的情况
    const float hX = static_cast<float>(maxX-minX)/nIni;

    // lNodes用于存放节点数据，note：只保留叶子节点
    // ExtractorNode中UL、UR、BL、BR记录了该节点（区域）的四个顶点坐标
    // ExtractorNode中的vKeys记录了属于该节点（区域）的所有特征点，这里有些低效，容器里存的是特征点而不是特征点的指针
    list<ExtractorNode> lNodes;

    // 记录初始节点的指针，为了方便根据特征点x坐标快速找到对应的节点（x/hX）
    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    // step1: 建立分裂的初始节点
    // step1.1：确定节点区域
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY); // wubo，为什么要减去minY
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    // step1.2：将所有特征点关联到对应的节点（区域）
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)    // 如果这个区域只有一个特征点，则不用再构建子树
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty()) // 如果这个区域一个特征点都没有，则删除该空节点
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    // 利用四叉树方法对图像进行划分区域
    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        // step2：广度搜索的方式遍历所有节点，将目前的子区域进行划分
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

        // step3：Node数快接近要求数目时，优先对包含特征点比较多的区域进行划分
        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)     // 当再划分之后所有的Node数快接近要求数目时，优先对包含特征点比较多的区域进行划分
        {
            while(!bFinish)
            {
                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                // 对需要划分的部分进行排序, 即对兴趣点数较多的区域进行划分
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());

                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
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

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    // step4：保留每个区域响应值最大的一个兴趣点
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}
