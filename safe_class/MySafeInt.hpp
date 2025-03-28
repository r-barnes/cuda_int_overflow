// Richard's Safe Int Class
#define DEVICE

namespace rsi {

namespace detail {

enum class rsi_arith_behaviour {
    STANDARD_BEHAVIOUR,
    STANDARD_BUT_NOT_SAME_SIZE,
    SAME_SIZE_PROMOTE_TO_LARGER_SIGNED,
    NONE
};

enum class rsi_accum_behaviour {
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL,
    ONLY_SMALLER,
    NONE
};

constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_ADD = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;
constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_ADD = rsi_accum_behaviour::STANDARD_BEHAVIOUR;
constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_SUB = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;
constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_SUB = rsi_accum_behaviour::STANDARD_BEHAVIOUR;
constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_DIV = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;
constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_DIV = rsi_accum_behaviour::STANDARD_BEHAVIOUR;
constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_MUL = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;
constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_MUL = rsi_accum_behaviour::STANDARD_BEHAVIOUR;
constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_MOD = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;
constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_MOD = rsi_accum_behaviour::STANDARD_BEHAVIOUR;

template<class T> struct LargerType {};
template<> struct LargerType<bool> { using type = int32_t; };
template<> struct LargerType<int8_t> { using type = int32_t; };
template<> struct LargerType<uint8_t> { using type = int32_t; };
template<> struct LargerType<int16_t> { using type = int32_t; };
template<> struct LargerType<uint16_t> { using type = int32_t; };
template<> struct LargerType<int32_t> { using type = int64_t; };
template<> struct LargerType<uint32_t> { using type = int64_t; };
template<> struct LargerType<int64_t> { using type = int64_t; };
template<> struct LargerType<uint64_t> { using type = int64_t; };

}

template <typename T>
class SafeInt {
  private:
    static_assert(std::is_integral_v<T>, "SafeInt can only hold integers");
    T m_value = 0;

  public:

    constexpr SafeInt() = default;
    DEVICE constexpr explicit SafeInt(T v) : m_value(v) {}

    explicit DEVICE constexpr operator T() const { return m_value; }

    T value() const { return m_value; }
    T& value() { return m_value; }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>(const SafeInt<U> &o){
        m_value = o.m_value;
    }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>& operator=(const SafeInt<U> &o){
        m_value = o.m_value;
        return *this;
    }

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>& operator=(const U &o){
        m_value = o;
        return *this;
    }  
};

}


// ADD arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator+(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_ADD == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_ADD == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing ADD (+) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_ADD == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) + static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_ADD == rsi::detail::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing ADD is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator+(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs + rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator+(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs + rsi::SafeInt(lhs);
}

// ADD accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator+=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_ADD == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_ADD == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing ADD accumulation (+=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_ADD == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing ADD accumulation (+=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_ADD == rsi::detail::rsi_accum_behaviour::NONE){
        static_assert(false, "Performing ADD is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator+=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += SafeInt(rhs);
    return lhs;
}


// SUB arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator-(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_SUB == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_SUB == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing SUB (-) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_SUB == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) - static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_SUB == rsi::detail::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing SUB is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator-(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs - rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator-(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs - rsi::SafeInt(lhs);
}

// SUB accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator-=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_SUB == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_SUB == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing SUB accumulation (-=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_SUB == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing SUB accumulation (-=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_SUB == rsi::detail::rsi_accum_behaviour::NONE){
        static_assert(false, "Performing SUB is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator-=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += SafeInt(rhs);
    return lhs;
}


// DIV arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator/(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_DIV == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_DIV == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing DIV (/) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_DIV == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) / static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_DIV == rsi::detail::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing DIV is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator/(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs / rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator/(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs / rsi::SafeInt(lhs);
}

// DIV accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator/=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_DIV == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_DIV == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing DIV accumulation (/=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_DIV == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing DIV accumulation (/=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_DIV == rsi::detail::rsi_accum_behaviour::NONE){
        static_assert(false, "Performing DIV is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator/=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += SafeInt(rhs);
    return lhs;
}


// MUL arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator*(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MUL == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MUL == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing MUL (*) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MUL == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) * static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MUL == rsi::detail::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing MUL is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator*(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs * rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator*(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs * rsi::SafeInt(lhs);
}

// MUL accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator*=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MUL == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MUL == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing MUL accumulation (*=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MUL == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing MUL accumulation (*=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MUL == rsi::detail::rsi_accum_behaviour::NONE){
        static_assert(false, "Performing MUL is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator*=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += SafeInt(rhs);
    return lhs;
}


// MOD arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator%(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MOD == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MOD == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing MOD (%) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MOD == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) % static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_MOD == rsi::detail::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing MOD is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator%(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs % rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator%(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs % rsi::SafeInt(lhs);
}

// MOD accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator%=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MOD == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MOD == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing MOD accumulation (%=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MOD == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing MOD accumulation (%=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_MOD == rsi::detail::rsi_accum_behaviour::NONE){
        static_assert(false, "Performing MOD is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator%=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += SafeInt(rhs);
    return lhs;
}

