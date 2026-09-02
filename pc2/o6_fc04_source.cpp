#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>


namespace {

std::atomic<bool> running{true};

void stop(int) {
    running = false;
}

uint64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint16_t crc16(const uint8_t* data, size_t length) {
    uint16_t crc = 0xffff;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1) ? (crc >> 1) ^ 0xa001 : crc >> 1;
        }
    }
    return crc;
}

class Fc04Port {
public:
    Fc04Port(const char* path, uint8_t slave_id) : path_(path), slave_id_(slave_id) {
        fd_ = ::open(path, O_RDWR | O_NOCTTY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error(path_ + ": open failed: " + std::strerror(errno));
        }
        if (::ioctl(fd_, TIOCEXCL) != 0) {
            close();
            throw std::runtime_error(path_ + ": exclusive access failed: " + std::strerror(errno));
        }
        termios config{};
        if (::tcgetattr(fd_, &config) != 0) {
            close();
            throw std::runtime_error(path_ + ": tcgetattr failed");
        }
        ::cfmakeraw(&config);
        if (::cfsetispeed(&config, B4000000) != 0 || ::cfsetospeed(&config, B4000000) != 0) {
            close();
            throw std::runtime_error(path_ + ": 4 Mbps setup failed");
        }
        config.c_cflag |= CLOCAL | CREAD;
        config.c_cflag &= ~CSTOPB;
        config.c_cflag &= ~PARENB;
        config.c_cflag &= ~CSIZE;
        config.c_cflag |= CS8;
        config.c_cc[VMIN] = 0;
        config.c_cc[VTIME] = 0;
        if (::tcsetattr(fd_, TCSANOW, &config) != 0 || ::tcflush(fd_, TCIOFLUSH) != 0) {
            close();
            throw std::runtime_error(path_ + ": serial configuration failed");
        }
    }

    Fc04Port(const Fc04Port&) = delete;
    Fc04Port& operator=(const Fc04Port&) = delete;

    ~Fc04Port() {
        close();
    }

    std::vector<int> read_input_registers(uint16_t start, uint16_t count) {
        std::array<uint8_t, 8> request{
            slave_id_, 0x04,
            static_cast<uint8_t>(start >> 8), static_cast<uint8_t>(start),
            static_cast<uint8_t>(count >> 8), static_cast<uint8_t>(count),
            0, 0,
        };
        const uint16_t request_crc = crc16(request.data(), 6);
        request[6] = static_cast<uint8_t>(request_crc);
        request[7] = static_cast<uint8_t>(request_crc >> 8);
        ::tcflush(fd_, TCIFLUSH);
        size_t sent = 0;
        while (sent < request.size()) {
            const ssize_t result = ::write(fd_, request.data() + sent, request.size() - sent);
            if (result <= 0) {
                throw std::runtime_error(path_ + ": FC04 query failed");
            }
            sent += static_cast<size_t>(result);
        }
        ::tcdrain(fd_);

        const size_t response_size = 5 + static_cast<size_t>(count) * 2;
        std::vector<uint8_t> response(response_size);
        size_t received = 0;
        while (received < response.size()) {
            pollfd descriptor{fd_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 100);
            if (ready <= 0 || (descriptor.revents & POLLIN) == 0) {
                throw std::runtime_error(path_ + ": FC04 response timeout");
            }
            const ssize_t result = ::read(fd_, response.data() + received, response.size() - received);
            if (result <= 0) {
                throw std::runtime_error(path_ + ": FC04 response failed");
            }
            received += static_cast<size_t>(result);
        }
        const uint16_t response_crc = crc16(response.data(), response.size() - 2);
        const uint16_t received_crc = response[response.size() - 2]
            | static_cast<uint16_t>(response[response.size() - 1]) << 8;
        if (response_crc != received_crc || response[0] != slave_id_
            || response[1] != 0x04 || response[2] != count * 2) {
            throw std::runtime_error(path_ + ": FC04 response rejected");
        }
        std::vector<int> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<int>(response[3 + i * 2]) << 8 | response[4 + i * 2];
        }
        return values;
    }

private:
    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::string path_;
    uint8_t slave_id_;
    int fd_ = -1;
};

struct Sample {
    uint64_t started_ns;
    uint64_t completed_ns;
    std::vector<int> registers;
};

Sample sample(Fc04Port& port) {
    Sample result;
    result.started_ns = monotonic_ns();
    result.registers = port.read_input_registers(0, 45);
    result.completed_ns = monotonic_ns();
    return result;
}

std::string version(const std::vector<int>& values, size_t index) {
    return std::to_string(values[index]) + "." + std::to_string(values[index + 1])
        + "." + std::to_string(values[index + 2]);
}

void print_six(const std::vector<int>& values, size_t start) {
    std::cout << '[';
    for (size_t i = 0; i < 6; ++i) {
        if (i) std::cout << ',';
        std::cout << values[start + i];
    }
    std::cout << ']';
}

void print_hand(const char* path, int slave_id, int direction, const Sample& sample) {
    const auto& values = sample.registers;
    std::cout << "{\"device_path\":\"" << path << "\",\"slave_id\":" << slave_id
              << ",\"direction_code\":" << direction
              << ",\"read_started_monotonic_ns\":" << sample.started_ns
              << ",\"read_completed_monotonic_ns\":" << sample.completed_ns
              << ",\"read_ok\":true,\"crc_valid\":true,\"freedom\":" << values[30]
              << ",\"version\":" << values[31]
              << ",\"device_number\":\"" << values[32] << '.' << values[33] << '.' << values[34]
              << "\",\"hardware_version\":\"" << version(values, 36)
              << "\",\"software_version\":\"" << version(values, 39)
              << "\",\"mechanical_version\":\"" << version(values, 42) << "\",\"angles_raw\":";
    print_six(values, 0);
    std::cout << ",\"torques_raw\":";
    print_six(values, 6);
    std::cout << ",\"speeds_raw\":";
    print_six(values, 12);
    std::cout << ",\"temperatures_raw\":";
    print_six(values, 18);
    std::cout << ",\"errors_raw\":";
    print_six(values, 24);
    std::cout << '}';
}

std::string read_boot_id() {
    std::ifstream stream("/proc/sys/kernel/random/boot_id");
    std::string value;
    std::getline(stream, value);
    if (value.empty()) throw std::runtime_error("source boot ID is unavailable");
    return value;
}

}  // namespace


int main(int argc, char** argv) {
    int samples = 0;
    double frequency = 30.0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--samples" && index + 1 < argc) {
            samples = std::stoi(argv[++index]);
        } else if (argument == "--frequency" && index + 1 < argc) {
            frequency = std::stod(argv[++index]);
        } else {
            std::cerr << "usage: o6_fc04_source [--samples N] [--frequency HZ]\n";
            return 2;
        }
    }
    if (samples < 0 || !std::isfinite(frequency) || frequency <= 0.0 || frequency > 100.0) {
        std::cerr << "invalid source arguments\n";
        return 2;
    }
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    try {
        Fc04Port left("/dev/ttyHAND0", 40);
        Fc04Port right("/dev/ttyHAND1", 39);
        const std::string boot_id = read_boot_id();
        uint64_t sequence = 0;
        auto next_tick = std::chrono::steady_clock::now();
        while (running && (samples == 0 || static_cast<int>(sequence) < samples)) {
            const Sample left_sample = sample(left);
            const Sample right_sample = sample(right);
            if (left_sample.registers[35] != 76 || right_sample.registers[35] != 82) {
                throw std::runtime_error("O6 direction identity rejected");
            }
            ++sequence;
            const double pair_skew_ms = std::abs(
                static_cast<double>((left_sample.started_ns + left_sample.completed_ns) / 2)
                - static_cast<double>((right_sample.started_ns + right_sample.completed_ns) / 2)
            ) / 1e6;
            std::cout << "{\"schema\":\"linker_o6_fc04_pair_v1\""
                      << ",\"configuration_id\":\"r1_a7_dual_linker_o6_v1\""
                      << ",\"source_boot_id\":\"" << boot_id << "\",\"sequence\":" << sequence
                      << ",\"source_monotonic_ns\":" << monotonic_ns()
                      << ",\"pair_skew_ms\":" << pair_skew_ms
                      << ",\"actuation_enabled\":false,\"writes_enabled\":false"
                      << ",\"protocol\":{\"transport\":\"modbus_rtu\",\"baudrate\":4000000"
                      << ",\"function_code\":4,\"start_register\":0,\"register_count\":45}"
                      << ",\"hardware_axis_order\":[\"thumb_cmc_pitch\",\"thumb_cmc_yaw\""
                      << ",\"index_mcp_pitch\",\"middle_mcp_pitch\",\"ring_mcp_pitch\",\"pinky_mcp_pitch\"]"
                      << ",\"hands\":{\"left\":";
            print_hand("/dev/ttyHAND0", 40, 76, left_sample);
            std::cout << ",\"right\":";
            print_hand("/dev/ttyHAND1", 39, 82, right_sample);
            std::cout << "}}\n" << std::flush;
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / frequency));
            std::this_thread::sleep_until(next_tick);
        }
    } catch (const std::exception& error) {
        std::cerr << "[O6 FC04 SOURCE] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
