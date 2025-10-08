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
// Modified_Version

#define _WIN32_WINNT 0x0A00  // Windows 11

#include <iostream>
#include "include/httplib.h"
#include "include/json.hpp"

using json = nlohmann::json;

int main() {
    httplib::Server svr;

    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "📩 Raw body: " << req.body << std::endl; // Print raw request body

        try {
            json received = json::parse(req.body);

            std::cout << "📩 Received: " << received.dump() << std::endl;

            if (!received.contains("real_points")) {
                std::cout << "⚠️ Key 'real_points' missing" << std::endl;
            }
            if (!received.contains("centers")) {
                std::cout << "⚠️ Key 'centers' missing" << std::endl;
            }

            auto final_points = received.value("real_points", json::array());
            auto centers = received.value("centers", json::array());

            std::cout << "final_points: " << final_points.dump() << std::endl;
            std::cout << "centers: " << centers.dump() << std::endl;

            json response = {
                {"status", "Received"},
                {"final_points", final_points},
                {"centers", centers}
            };

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            std::cerr << "❌ Error parsing JSON: " << e.what() << std::endl;
            res.status = 400;
            res.set_content(R"({"status":"Error","error":"Invalid JSON"})", "application/json");
        }
    });

    std::cout << "🚀 C++ Receiver running on port 5000..." << std::endl;
    svr.listen("0.0.0.0", 5000);

    return 0;
}
