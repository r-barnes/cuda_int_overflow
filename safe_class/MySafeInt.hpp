#define DEVICE

template<class T> struct LargerType {};
template<> struct LargerType<int8_t> { using type = int16_t; };
template<> struct LargerType<uint8_t> { using type = int16_t; };
template<> struct LargerType<int16_t> { using type = int32_t; };
template<> struct LargerType<uint16_t> { using type = int32_t; };
template<> struct LargerType<int32_t> { using type = int64_t; };
template<> struct LargerType<uint32_t> { using type = int64_t; };
template<> struct LargerType<int64_t> { using type = int64_t; };
template<> struct LargerType<uint64_t> { using type = int64_t; };

template <typename T>
struct MySafeInt {
    static_assert(std::is_integral_v<T>, "MySafeInt can only hold integers");
    T m_value = 0;

    constexpr MySafeInt() = default;
    DEVICE constexpr explicit MySafeInt(T v) : m_value(v) {}

    explicit DEVICE constexpr operator T() const { return m_value; }

    T value() const { return m_value; }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>(const MySafeInt<U> &o){
        m_value = o.m_value;
    }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>& operator=(const MySafeInt<U> &o){
        m_value = o.m_value;
        return *this;
    }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>& operator=(const U &o){
        m_value = o;
        return *this;
    }  
};


// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator+(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value + rhs.m_value);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator+(const MySafeInt<T>& lhs, const U& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value + rhs);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator+(const U& lhs, const MySafeInt<T>& rhs) {
    return rhs + lhs;
}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto& operator+=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    lhs.m_value += rhs.m_value;
    return lhs;
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto operator+=(MySafeInt<T>& lhs, const U& rhs) {
    lhs.m_value += rhs;
    return lhs;
}


// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator-(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value - rhs.m_value);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator-(const MySafeInt<T>& lhs, const U& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value - rhs);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator-(const U& lhs, const MySafeInt<T>& rhs) {
    return rhs - lhs;
}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto& operator-=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    lhs.m_value += rhs.m_value;
    return lhs;
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto operator-=(MySafeInt<T>& lhs, const U& rhs) {
    lhs.m_value += rhs;
    return lhs;
}


// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator/(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value / rhs.m_value);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator/(const MySafeInt<T>& lhs, const U& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value / rhs);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator/(const U& lhs, const MySafeInt<T>& rhs) {
    return rhs / lhs;
}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto& operator/=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    lhs.m_value += rhs.m_value;
    return lhs;
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto operator/=(MySafeInt<T>& lhs, const U& rhs) {
    lhs.m_value += rhs;
    return lhs;
}


// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator*(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    if constexpr (sizeof(T) != sizeof(U)){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value * rhs.m_value);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator*(const MySafeInt<T>& lhs, const U& rhs) {
    if constexpr (sizeof(T) != sizeof(U)){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value * rhs);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator*(const U& lhs, const MySafeInt<T>& rhs) {
    return rhs * lhs;
}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && sizeof(T) != sizeof(U)>>
DEVICE constexpr auto& operator*=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    lhs.m_value += rhs.m_value;
    return lhs;
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && sizeof(T) != sizeof(U)>>
DEVICE constexpr auto operator*=(MySafeInt<T>& lhs, const U& rhs) {
    lhs.m_value += rhs;
    return lhs;
}


// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator%(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value % rhs.m_value);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator%(const MySafeInt<T>& lhs, const U& rhs) {
    if constexpr (true){
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value % rhs);
    } else {
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator%(const U& lhs, const MySafeInt<T>& rhs) {
    return rhs % lhs;
}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto& operator%=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {
    lhs.m_value += rhs.m_value;
    return lhs;
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && true>>
DEVICE constexpr auto operator%=(MySafeInt<T>& lhs, const U& rhs) {
    lhs.m_value += rhs;
    return lhs;
}

