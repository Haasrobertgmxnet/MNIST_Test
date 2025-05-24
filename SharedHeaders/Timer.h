#pragma once

#include <chrono>
#include <print>

namespace Helper {
    class Timer
    {
    public:
        Timer() : outputAtExit(true)
        {
            start = std::chrono::system_clock::now();
        }

        void setOutputAtExit(bool _outputAtExit) {
            outputAtExit = _outputAtExit;
        }

        std::chrono::system_clock::time_point getStart() const {
            return start;
        }

        std::chrono::milliseconds getDuration() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - start);
        }

        ~Timer()
        {
            if (outputAtExit)
            {
                std::print(
                    "Time difference needed for program execution: {} Milliseconds.\n",
                    getDuration().count());
            }
        }

    private:
        bool outputAtExit{};
        std::chrono::system_clock::time_point start{};
    };
}
