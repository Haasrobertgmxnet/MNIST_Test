#pragma once

#include <filesystem>

namespace fs = std::filesystem;

namespace Helper {
    class PathNameService {
    public:
        static std::optional<std::string>
            findFileInCurrentDirectory(const std::string& filename)
        {
            fs::path currentDir = fs::current_path() / "..";

            for (const auto& entry : fs::directory_iterator(currentDir))
            {
                if (entry.is_regular_file() && entry.path().filename() == filename)
                {
                    // Rückgabe: relativer Pfad zum Arbeitsverzeichnis
                    return entry.path().string();
                    // Oder absolut:
                    // return fs::absolute(entry.path()).string();
                }
            }

            // Datei nicht gefunden
            return std::nullopt;
        }

        static std::optional<std::string>
            findFileAboveCurrentDirectory(const std::string& filename)
        {
            // Starte im übergeordneten Verzeichnis
            fs::path startDir = fs::current_path().parent_path();

            // Durchsuche rekursiv alle Unterverzeichnisse
            for (const auto& entry : fs::recursive_directory_iterator(startDir))
            {
                auto tmp{ entry.path().filename().string() };
                if (entry.path().filename().string() == filename)
                {
                    return entry.path()
                        .string(); // Oder fs::absolute(entry.path()).string()
                }
            }

            return std::nullopt; // Datei nicht gefunden
        }
    };
}