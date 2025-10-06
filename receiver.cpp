// #include <httplib.h>
// #include <nlohmann/json.hpp>
// #include <iostream>

// using json = nlohmann::json;

// int main() {
//     httplib::Server svr;

//     svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
//         try {
//             json data = json::parse(req.body);
//             std::cout << "📩 Received: " << data.dump(4) << std::endl;

//             json response = { {"status", "Received"}, {"data", data} };
//             res.set_content(response.dump(), "application/json");
//         } catch (std::exception& e) {
//             json error = { {"status", "Error"}, {"error", e.what()} };
//             res.status = 400;
//             res.set_content(error.dump(), "application/json");
//         }
//     });

//     std::cout << "🚀 Receiver running on port 5000..." << std::endl;
//     svr.listen("0.0.0.0", 5000);
// }

////////////
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using json = nlohmann::json;

// Compact receiver function
void startReceiver(int port = 5000) {
    httplib::Server svr;

    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        try {
            json data = json::parse(req.body);

            // 📩 Print received JSON
            std::cout << "📩 Received JSON:\n" << data.dump(4) << std::endl;

            // Example: Access specific fields
            if (data.contains("centers")) {
                auto centers = data["centers"];
                std::cout << "✅ Centers detected: " << centers.size() << std::endl;
            }
            if (data.contains("real_points")) {
                auto points = data["real_points"];
                std::cout << "✅ Real points detected: " << points.size() << std::endl;
            }

            // Respond to sender
            json response = { {"status", "Received"}, {"data", data} };
            res.set_content(response.dump(), "application/json");

            // ⚡ Here you can call your robot functions instead of just printing
            // e.g. moveRobotArm(centers, points);

        } catch (std::exception& e) {
            json error = { {"status", "Error"}, {"error", e.what()} };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
        }
    });

    std::cout << "🚀 Receiver running on port " << port << "..." << std::endl;
    svr.listen("0.0.0.0", port);
}

// Example usage with robot main loop
int main() {
    std::thread receiverThread([]() {
        startReceiver(5000); // run HTTP receiver
    });
    receiverThread.detach(); // keep server running in background

    // Your robot's main logic
    while (true) {
        std::cout << "🤖 Robot main loop running..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    return 0;
}
