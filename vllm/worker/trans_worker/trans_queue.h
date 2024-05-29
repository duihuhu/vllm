#include <deque>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template <typename T>
class TransQueue {
private:
    std::deque<T> deque_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;

public:
    // Push element to the front of the deque
    void push_front(const T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.push_front(value);
        cv_.notify_one();
    }

    // Push element to the back of the deque
    void push_back(const T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.push_back(value);
        cv_.notify_one();
    }

    // Pop element from the front of the deque
    T pop_front() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !deque_.empty(); });

        T value = deque_.front();
        deque_.pop_front();
        return value;
    }

    // Pop element from the back of the deque
    T pop_back() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !deque_.empty(); });

        T value = deque_.back();
        deque_.pop_back();
        return value;
    }

    // Check if the deque is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.empty();
    }

    // Get the size of the deque
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.size();
    }
};

