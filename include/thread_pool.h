#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool;

struct WorkerData {
    ThreadPool* pool;
};

void worker_function(WorkerData* data);

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<void (*)()> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

    friend void worker_function(ThreadPool* pool);

public:
    ThreadPool(size_t num_threads = 4) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.push_back(std::thread(worker_function, this));
        }
    }

private:
    void worker_thread() {
        while (true) {
            void (*task)() = nullptr;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this] {
                    return stop || !tasks.empty();
                });

                if (stop && tasks.empty()) {
                    return;
                }

                task = tasks.front();
                tasks.pop();
            }
            if (task) {
                task();
            }
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        for (size_t i = 0; i < workers.size(); ++i) {
            if (workers[i].joinable()) {
                workers[i].join();
            }
        }
    }

    void enqueue(void (*task)()) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                return;
            }
            tasks.push(task);
        }
        condition.notify_one();
    }

    size_t size() const {
        return workers.size();
    }
};

void worker_function(ThreadPool* pool) {
    while (true) {
        void (*task)() = nullptr;
        {
            std::unique_lock<std::mutex> lock(pool->queue_mutex);
            while (!pool->stop && pool->tasks.empty()) {
                pool->condition.wait(lock);
            }

            if (pool->stop && pool->tasks.empty()) {
                return;
            }

            task = pool->tasks.front();
            pool->tasks.pop();
        }
        if (task) {
            task();
        }
    }
}