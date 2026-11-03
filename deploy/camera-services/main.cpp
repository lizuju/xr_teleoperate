// Sets the head camera services after the robot boots, because the app's switches do not survive a restart.
//
// Usage: r1_camera_services INTERFACE list
//        r1_camera_services INTERFACE apply [SETTLE_SECONDS]
//
// apply watches for SETTLE_SECONDS after the service list first answers, since services
// can start late in boot and undo an early switch. It then exits, so the app can
// still change these services for the rest of the session.
#include <unitree/robot/b2/robot_state/robot_state_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include <chrono>
#include <cstdio>
#include <exception>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace b2 = unitree::robot::b2;
using Clock = std::chrono::steady_clock;

namespace {

// RobotStateClient uses opposite numbers for the two directions (Unitree docs):
// the switch request is 1 = on, 0 = off, while reported status is 0 = on, 1 = off.
constexpr int32_t kSwitchOn = 1, kSwitchOff = 0;
constexpr int32_t kStatusOn = 0;

struct Wanted {
    const char* name;
    bool on;
};
// Release the camera before asking for it.
const std::vector<Wanted> kWanted = {{"video_hub", false}, {"stereo_patch_pc1", true}};

constexpr auto kPollPeriod = std::chrono::seconds(5);
constexpr auto kFirstAnswerLimit = std::chrono::minutes(10);

bool Query(b2::RobotStateClient& client, std::map<std::string, b2::ServiceState>& services) {
    std::vector<b2::ServiceState> list;
    int32_t code = client.ServiceList(list);
    if (code != 0) {
        std::printf("[camera-services] service list failed: code=%d\n", code);
        return false;
    }
    services.clear();
    for (const auto& service : list) services[service.name] = service;
    return true;
}

// True when every wanted service is present and in the wanted state.
bool Enforce(b2::RobotStateClient& client, const std::map<std::string, b2::ServiceState>& services) {
    bool settled = true;
    for (const auto& wanted : kWanted) {
        auto found = services.find(wanted.name);
        if (found == services.end()) {
            std::printf("[camera-services] %s is not listed yet\n", wanted.name);
            settled = false;
            continue;
        }
        if ((found->second.status == kStatusOn) == wanted.on) continue;
        settled = false;
        int32_t status = -1;
        int32_t code = client.ServiceSwitch(wanted.name, wanted.on ? kSwitchOn : kSwitchOff, status);
        std::printf("[camera-services] %s reported status=%d, switching %s: code=%d status=%d\n", wanted.name,
                    found->second.status, wanted.on ? "on" : "off", code, status);
    }
    return settled;
}

int List(b2::RobotStateClient& client) {
    std::map<std::string, b2::ServiceState> services;
    if (!Query(client, services)) return 1;
    for (const auto& entry : services) {
        std::printf("%-32s %-3s protect=%d\n", entry.first.c_str(), entry.second.status == kStatusOn ? "on" : "off",
                    entry.second.protect);
    }
    return 0;
}

int Apply(b2::RobotStateClient& client, int settle_seconds) {
    const auto started = Clock::now();
    Clock::time_point first_answer{};
    bool answered = false;
    while (true) {
        const auto now = Clock::now();
        if (answered && now - first_answer >= std::chrono::seconds(settle_seconds)) break;
        if (!answered && now - started >= kFirstAnswerLimit) {
            std::printf("[camera-services] no service list after %d s\n", static_cast<int>(
                std::chrono::duration_cast<std::chrono::seconds>(kFirstAnswerLimit).count()));
            return 1;
        }
        std::map<std::string, b2::ServiceState> services;
        if (Query(client, services)) {
            if (!answered) {
                answered = true;
                first_answer = now;
                std::printf("[camera-services] service list answered after %d s\n", static_cast<int>(
                    std::chrono::duration_cast<std::chrono::seconds>(now - started).count()));
            }
            Enforce(client, services);
        }
        std::fflush(stdout);
        std::this_thread::sleep_for(kPollPeriod);
    }
    // The last switch may have been sent on the last pass, so judge from a fresh list.
    std::map<std::string, b2::ServiceState> services;
    const bool settled = Query(client, services) && Enforce(client, services);
    std::printf("[camera-services] %s\n", settled ? "done: video_hub off, stereo_patch_pc1 on" : "NOT settled");
    return settled ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string mode = argc >= 3 ? argv[2] : "";
    if (mode != "list" && mode != "apply") {
        std::fprintf(stderr, "Usage: %s INTERFACE list | %s INTERFACE apply [SETTLE_SECONDS]\n", argv[0], argv[0]);
        return 2;
    }
    try {
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
        b2::RobotStateClient client;
        client.SetTimeout(5.0f);
        client.Init();
        return mode == "list" ? List(client) : Apply(client, argc >= 4 ? std::stoi(argv[3]) : 180);
    } catch (const std::exception& error) {
        std::printf("[camera-services] %s\n", error.what());
        return 1;
    }
}
