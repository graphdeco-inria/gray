find cuda \
  -path "*/third_party/*" -prune -o \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i --style="{SortIncludes: false, ColumnLimit: 120, IndentWidth: 4, UseTab: Never, IncludeBlocks: Preserve}" {} +
ruff format *.py gray/
