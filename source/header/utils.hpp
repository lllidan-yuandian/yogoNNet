//
// Created by yogo on 18-9-27.
//

#ifndef YOGONNET_UTILS_H
#define YOGONNET_UTILS_H

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif //YOGONNET_UTILS_H
