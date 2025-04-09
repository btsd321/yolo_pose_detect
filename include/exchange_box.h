#ifndef EXCHANGE_BOX_H
#define EXCHANGE_BOX_H

#include <opencv2/opencv.hpp>

class ExchangeBox
{
public:
    ExchangeBox(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4, float s, cv::Rect b)
    : poses_{p1, p2, p3, p4}, score_(s), box_(b)
    {}
    cv::Point2f get_point(int index) const
    {
        return poses_.at(index);
    }
    float get_score() const
    {
        return score_;
    }
    cv::Rect get_box() const
    {
        return box_;
    }
private:
    std::array<cv::Point2f, 4> poses_;
    float score_=0;
    cv::Rect box_;
};
   



#endif 

