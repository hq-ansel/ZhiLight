#pragma once
#include "easyengine/core/export.hh"
#include "3rd/spdlog/spdlog.h"
#include <string>
#include <sstream>
namespace easyengine {
namespace core {

enum class LogLevel { kLogInfo, kLogWarning, kLogError, kLogDebug, kLogCritical, kLogOff };

class LogLine;
class ENGINE_EXPORT Logger {
public:
    virtual ~Logger() = default;
    virtual void info(const std::string& message) = 0;
    virtual void warn(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void debug(const std::string& message) = 0;
    virtual void critical(const std::string& message) = 0;
    virtual void set_log_level(LogLevel level) = 0;

    LogLine info();
    LogLine warn();
    LogLine error();
    LogLine debug();
    LogLine critical();
};

class ENGINE_EXPORT LoggerFactory {
public:
    virtual ~LoggerFactory() = default;
    virtual Logger* create_logger(const std::string& name) = 0;
    virtual void set_log_level(LogLevel level) = 0;
};

class ENGINE_EXPORT LogLine {
private:
    Logger* logger;
    int lvl;
    std::stringstream ss;

public:
    LogLine(Logger* logger, int lvl);
    ~LogLine() noexcept;
    LogLine(const LogLine&) = delete;
    LogLine(LogLine&& other);
    template<typename T>
    LogLine& operator<<(const T& t) {
        ss << t;
        return *this;
    }
};
class StandardLogger : public core::Logger {
    std::shared_ptr<spdlog::logger> logger;

public:
    StandardLogger(std::shared_ptr<spdlog::logger> logger) : logger(logger) {
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    }
    ~StandardLogger() = default;

    void set_log_level(core::LogLevel level) override {
        switch (level) {
            case core::LogLevel::kLogDebug: logger->set_level(spdlog::level::debug); break;
            case core::LogLevel::kLogInfo: logger->set_level(spdlog::level::info); break;
            case core::LogLevel::kLogWarning: logger->set_level(spdlog::level::warn); break;
            case core::LogLevel::kLogError: logger->set_level(spdlog::level::err); break;
            case core::LogLevel::kLogCritical: logger->set_level(spdlog::level::critical); break;
            default: logger->set_level(spdlog::level::off); break;
        }
    }

    void info(const std::string& message) noexcept override { logger->info(message); }
    void warn(const std::string& message) noexcept override { logger->warn(message); }
    void error(const std::string& message) noexcept override { logger->error(message); }
    void debug(const std::string& message) noexcept override { logger->debug(message); }
    void critical(const std::string& message) noexcept override { logger->critical(message); }
};
}
}