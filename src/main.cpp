#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "cli/cli_options.hpp"
#include "cli/processing.hpp"

int main(int argc, char** argv) {
    try {
        const tgpu::cli::CliArguments arguments = tgpu::cli::parse_cli_arguments(argc, argv);
        if (std::filesystem::is_directory(arguments.input_path)) {
            return tgpu::cli::run_directory_mode(arguments);
        }
        return tgpu::cli::run_single_file_mode(arguments);
    } catch (const std::invalid_argument&) {
        tgpu::cli::print_usage(std::cout);
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
