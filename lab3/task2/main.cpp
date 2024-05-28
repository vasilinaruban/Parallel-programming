#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <functional>

std::mutex m;
std::condition_variable cv;

std::queue<std::pair<size_t, std::function<double()>>> task_queue;
bool is_running = false;

struct Task {
    int type;
    double arg1;
    double arg2; 
};

template <typename T>
class Server {
public:
    Server() : is_running(false), next_task_id(0) {}

    void start() {
        if (!is_running) {
        	std::unique_lock<std::mutex> lock(m);
            is_running = true;
            server_thread = std::thread(&Server::process_tasks, this);
        }
    }

    void stop() {
        if (is_running) {
        	std::unique_lock<std::mutex> lock(m);
            is_running = false;
            cv.notify_all();
            lock.unlock();
            server_thread.join();
        }
    }

    size_t add_task(Task task) {
        std::unique_lock<std::mutex> lock(m);
        size_t task_id = next_task_id++;
        task_queue.push({task_id, create_task_function(task)});
        cv.notify_all();
        return task_id;
    }

    T request_result(size_t id_res) {
        std::unique_lock<std::mutex> lock(m);
        cv_result.wait(lock, [&]{ return results.find(id_res) != results.end(); });
        T result = results[id_res];
        results.erase(id_res);
        return result;
    }

private:
    std::thread server_thread;
    bool is_running;
    size_t next_task_id;
    std::unordered_map<size_t, T> results;

    void process_tasks() {
        while (is_running) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&]{ return !task_queue.empty() || !is_running; });
            if (!is_running && task_queue.empty()) return;

            auto [task_id, task_func] = task_queue.front();
            task_queue.pop();
            lock.unlock();

            T result = task_func();
            lock.lock();
            results[task_id] = result;
            cv_result.notify_all();
        }
    }

    std::function<T()> create_task_function(const Task& task) {
        switch (task.type) {
            case 1:
                return [arg = task.arg1]() { return std::sin(arg); };
            case 2:
                return [arg = task.arg1]() { return std::sqrt(arg); };
            case 3:
                return [base = task.arg1, exp = task.arg2]() { return std::pow(base, exp); };
            default:
                throw std::invalid_argument("Unknown task type");
        }
    }

    std::condition_variable cv_result;
};

void client_function(Server<double>& server, int task_type, int num_tasks, const std::string& filename) {
    std::ofstream file(filename);

    std::vector<size_t> task_ids;
    for (int i = 0; i < num_tasks; ++i) {
        Task task;
        task.type = task_type;
        task.arg1 = 10.0 + i; 
        if (task_type == 3) {
            task.arg2 = 2.0 + i; 
        }
        size_t task_id = server.add_task(task);
        task_ids.push_back(task_id);
    }

    for (size_t task_id : task_ids) {
        double result = server.request_result(task_id);
        file << "Task ID: " << task_id << " Result: " << result << "\n";
    }
}

int main() {
    Server<double> server;
    server.start();

    const int num_tasks = 10;

    std::thread client1(client_function, std::ref(server), 1, num_tasks, "results_sin.txt");
    std::thread client2(client_function, std::ref(server), 2, num_tasks, "results_sqrt.txt");
    std::thread client3(client_function, std::ref(server), 3, num_tasks, "results_pow.txt");

    client1.join();
    client2.join();
    client3.join();

    server.stop();

    return 0;
}
