#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

class FIFOWorkQueue {
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
    bool stop_ = false;

public:
    FIFOWorkQueue(size_t num_workers) {
        for (size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());
        auto task_ptr = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(f));
        auto future = task_ptr->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task_ptr]() { (*task_ptr)(); });
        }
        cv_.notify_one();
        return future;
    }

    ~FIFOWorkQueue() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
};