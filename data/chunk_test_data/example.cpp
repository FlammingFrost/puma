//
// 9.1.noexcept.cpp
// chapter 09 others
// modern c++ tutorial
//
// created by changkun at changkun.de
// https://github.com/changkun/modern-cpp-tutorial
//

#include <iostream>
#include <string>

void may_throw() {
    throw true;
}
auto non_block_throw = []{
    may_throw();
};
void no_throw() noexcept {
    return;
}

auto block_throw = []() noexcept {
    no_throw();
};

int main()
{
    std::cout << std::boolalpha
    << "may_throw() noexcept? " << noexcept(may_throw()) << std::endl
    << "no_throw() noexcept? " << noexcept(no_throw()) << std::endl
    << "lmay_throw() noexcept? " << noexcept(non_block_throw()) << std::endl
    << "lno_throw() noexcept? " << noexcept(block_throw()) << std::endl;
    
    try {
        may_throw();
    } catch (...) {
        std::cout << "exception captured from my_throw()" << std::endl;
    }
    
    try {
        non_block_throw();
    } catch (...) {
        std::cout << "exception captured from non_block_throw()" << std::endl;
    }
    
    try {
        block_throw();
    } catch (...) {
        std::cout << "exception captured from block_throw()" << std::endl;
    }
}

std::string operator"" _wow1(const char *wow1, size_t len) {
    return std::string(wow1)+"woooooooooow, amazing";
}

std::string operator""_wow2 (unsigned long long i) {
    return std::to_string(i)+"woooooooooow, amazing";
}

int main() {
    std::string str = R"(C:\\File\\To\\Path)";
    std::cout << str << std::endl;
    
    int value = 0b1001010101010;
    std::cout << value << std::endl;
    
    
    auto str2 = "abc"_wow1;
    auto num = 1_wow2;
    std::cout << str2 << std::endl;
    std::cout << num << std::endl;
    return 0;
}

struct Storage {
    char      a;
    int       b;
    double    c;
    long long d;
};

struct alignas(std::max_align_t) AlignasStorage {
    char      a;
    int       b;
    double    c;
    long long d;
};

int main() {
    std::cout << alignof(Storage) << std::endl;
    std::cout << alignof(AlignasStorage) << std::endl;
    return 0;
}