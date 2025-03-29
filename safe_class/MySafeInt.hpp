// Richard's Safe Int Class
#include <ostream>

#define RSI_DEVICE

#ifndef RSI_CONVERSION_TYPE
#define RSI_CONVERSION_TYPE // blank means implicit
#endif

#ifndef RSI_ALLOW_CONVERSION
#define RSI_ALLOW_CONVERSION 1
#endif

#ifndef RSI_ASSIGN_BEHAVIOUR
#define RSI_ASSIGN_BEHAVIOUR rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR
#endif
#ifndef RSI_ARITH_BEHAVIOUR_ADD
#define RSI_ARITH_BEHAVIOUR_ADD rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_ADD
#define RSI_ARITH_ACCUM_BEHAVIOUR_ADD rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_BEHAVIOUR_SUB
#define RSI_ARITH_BEHAVIOUR_SUB rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_SUB
#define RSI_ARITH_ACCUM_BEHAVIOUR_SUB rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_BEHAVIOUR_DIV
#define RSI_ARITH_BEHAVIOUR_DIV rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_DIV
#define RSI_ARITH_ACCUM_BEHAVIOUR_DIV rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_BEHAVIOUR_MUL
#define RSI_ARITH_BEHAVIOUR_MUL rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_MUL
#define RSI_ARITH_ACCUM_BEHAVIOUR_MUL rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_BEHAVIOUR_MOD
#define RSI_ARITH_BEHAVIOUR_MOD rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif

#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_MOD
#define RSI_ARITH_ACCUM_BEHAVIOUR_MOD rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif


namespace rsi {

namespace detail {

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

enum class rsi_arith_behaviour {
    STANDARD_BEHAVIOUR,
    STANDARD_BUT_NOT_SAME_SIZE,
    SAME_SIZE_PROMOTE_TO_LARGER_SIGNED,
    NONE
};

enum class rsi_arith_accum_behaviour {
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL,
    ONLY_SMALLER,
    NONE
};

enum class rsi_assign_behaviour {
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL_SAME_SIGN,
    SMALLER_OR_EQUAL,
    NONE
};

enum class rsi_bitwise_behaviour {
    STANDARD_BEHAVIOUR,
    STANDARD_BUT_ONLY_UNSIGNED,
    UNSIGNED_AND_SAME_SIZE,
    NONE
};

template <typename T>
class SafeInt {
  private:
    static_assert(std::is_integral_v<T>, "SafeInt can only hold integers");
    T m_value = 0;

  public:

    constexpr SafeInt() = default;
    RSI_DEVICE constexpr SafeInt(T v) : m_value(v) {}

    #if RSI_ALLOW_CONVERSION
    RSI_CONVERSION_TYPE RSI_DEVICE constexpr operator T() const { return m_value; }
    #endif

    T value() const { return m_value; }
    T& value() { return m_value; }

    template<typename U>
    SafeInt<T>(const SafeInt<U> &o){
        if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR) {
            m_value = o.m_value;
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL) {
            if constexpr (sizeof(T) >= sizeof(U)) {
                m_value = o.m_value;
            } else {
                static_assert(false, "Assignment not allowed: rhs type is larger than lhs type.");
            }
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL_SAME_SIGN) {
            if constexpr (sizeof(T) >= sizeof(U) && std::is_signed_v<T> == std::is_signed_v<U>) {
                m_value = o.m_value;
            } else {
                static_assert(false, "Assignment not allowed: rhs type is larger than lhs type.");
            }
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::NONE){
            static_assert(false, "Performing assignment is not allowed.");
        } else {
            static_assert(false, "Unrecognized behaviour.");
        }
    }

    template<typename U>
    SafeInt<T>& operator=(const SafeInt<U> &o){
        if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR) {
            m_value = o.m_value;
            return *this;
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL) {
            if constexpr (sizeof(T) >= sizeof(U)) {
                m_value = o.m_value;
                return *this;
            } else {
                static_assert(false, "Assignment not allowed: rhs type is larger than lhs type.");
            }
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL_SAME_SIGN) {
            if constexpr (sizeof(T) >= sizeof(U) && std::is_signed_v<T> == std::is_signed_v<U>) {
                m_value = o.m_value;
                return *this;
            } else {
                static_assert(false, "Assignment not allowed: rhs type is larger than lhs type.");
            }
        } else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::NONE){
            static_assert(false, "Performing assignment is not allowed.");
        } else {
            static_assert(false, "Unrecognized behaviour.");
        }
    }

    template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
    SafeInt<T>& operator=(const U &rhs){
        return (*this = SafeInt(rhs));
    }

    // prefix increment
    SafeInt<T>& operator++(){
        ++m_value;
        return *this;
    }
 
    // postfix increment
    SafeInt<T> operator++(int){
        auto old = *this;  // copy old value
        operator++();  // prefix increment
        return old;    // return old value
    }
};

}

template<typename T, typename U>
RSI_DEVICE constexpr auto operator<=>(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    return lhs.value() <=> rhs.value();
}

template<typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
RSI_DEVICE constexpr auto operator<=>(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs.value() <=> rhs;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const rsi::SafeInt<T>& x) {
   return out << x.value();
}


// ADD arithmetic operations
template<typename T, typename U>
RSI_DEVICE constexpr auto operator+(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_BEHAVIOUR_ADD == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (RSI_ARITH_BEHAVIOUR_ADD == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing ADD (+) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_ADD == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) + static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_ADD == rsi::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing ADD (+) is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator+(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs + rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator+(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs + rsi::SafeInt(lhs);
}

// ADD accumulation operators
template<typename T, typename U>
RSI_DEVICE constexpr auto& operator+=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_ADD == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() += rhs.value());
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_ADD == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing ADD accumulation (+=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_ADD == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() += rhs.value();
        } else {
            static_assert(false, "Performing ADD accumulation (+=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_ADD == rsi::rsi_arith_accum_behaviour::NONE){
        static_assert(false, "Performing ADD accumulation (+=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto& operator+=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs += rsi::SafeInt<U>(rhs);
    return lhs;
}


// SUB arithmetic operations
template<typename T, typename U>
RSI_DEVICE constexpr auto operator-(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_BEHAVIOUR_SUB == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (RSI_ARITH_BEHAVIOUR_SUB == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing SUB (-) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_SUB == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) - static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_SUB == rsi::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing SUB (-) is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator-(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs - rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator-(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs - rsi::SafeInt(lhs);
}

// SUB accumulation operators
template<typename T, typename U>
RSI_DEVICE constexpr auto& operator-=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_SUB == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() -= rhs.value());
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_SUB == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() -= rhs.value();
        } else {
            static_assert(false, "Performing SUB accumulation (-=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_SUB == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() -= rhs.value();
        } else {
            static_assert(false, "Performing SUB accumulation (-=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_SUB == rsi::rsi_arith_accum_behaviour::NONE){
        static_assert(false, "Performing SUB accumulation (-=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto& operator-=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs -= rsi::SafeInt<U>(rhs);
    return lhs;
}


// DIV arithmetic operations
template<typename T, typename U>
RSI_DEVICE constexpr auto operator/(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_BEHAVIOUR_DIV == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (RSI_ARITH_BEHAVIOUR_DIV == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing DIV (/) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_DIV == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) / static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_DIV == rsi::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing DIV (/) is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator/(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs / rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator/(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs / rsi::SafeInt(lhs);
}

// DIV accumulation operators
template<typename T, typename U>
RSI_DEVICE constexpr auto& operator/=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_DIV == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() /= rhs.value());
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_DIV == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() /= rhs.value();
        } else {
            static_assert(false, "Performing DIV accumulation (/=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_DIV == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() /= rhs.value();
        } else {
            static_assert(false, "Performing DIV accumulation (/=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_DIV == rsi::rsi_arith_accum_behaviour::NONE){
        static_assert(false, "Performing DIV accumulation (/=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto& operator/=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs /= rsi::SafeInt<U>(rhs);
    return lhs;
}


// MUL arithmetic operations
template<typename T, typename U>
RSI_DEVICE constexpr auto operator*(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_BEHAVIOUR_MUL == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MUL == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing MUL (*) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MUL == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) * static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MUL == rsi::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing MUL (*) is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator*(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs * rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator*(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs * rsi::SafeInt(lhs);
}

// MUL accumulation operators
template<typename T, typename U>
RSI_DEVICE constexpr auto& operator*=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MUL == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() *= rhs.value());
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MUL == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() *= rhs.value();
        } else {
            static_assert(false, "Performing MUL accumulation (*=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MUL == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() *= rhs.value();
        } else {
            static_assert(false, "Performing MUL accumulation (*=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MUL == rsi::rsi_arith_accum_behaviour::NONE){
        static_assert(false, "Performing MUL accumulation (*=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto& operator*=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs *= rsi::SafeInt<U>(rhs);
    return lhs;
}


// MOD arithmetic operations
template<typename T, typename U>
RSI_DEVICE constexpr auto operator%(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_BEHAVIOUR_MOD == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {
        return rsi::SafeInt(lhs.value() * rhs.value());
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MOD == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {
        if constexpr (sizeof(T) == sizeof(U)) {
            static_assert(false, "Performing MOD (%) on types of the same size is not allowed.");
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MOD == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){
        if constexpr (sizeof(T) == sizeof(U)) {
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) % static_cast<larger>(rhs.value()));
        } else {
            return rsi::SafeInt(lhs.value() * rhs.value());
        }
    } else if constexpr (RSI_ARITH_BEHAVIOUR_MOD == rsi::rsi_arith_behaviour::NONE){
        static_assert(false, "Performing MOD (%) is not allowed.");
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator%(const rsi::SafeInt<T>& lhs, const U& rhs) {
    return lhs % rsi::SafeInt(rhs);
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto operator%(const U& lhs, const rsi::SafeInt<T>& rhs) {
    return rhs % rsi::SafeInt(lhs);
}

// MOD accumulation operators
template<typename T, typename U>
RSI_DEVICE constexpr auto& operator%=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MOD == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {
        return (lhs.value() %= rhs.value());
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MOD == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {
        if constexpr (sizeof(T) >= sizeof(U)) {
            return lhs.value() %= rhs.value();
        } else {
            static_assert(false, "Performing MOD accumulation (%=) with the right-hand side of a larger type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MOD == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){
        if constexpr (sizeof(T) > sizeof(U)) {
            return lhs.value() %= rhs.value();
        } else {
            static_assert(false, "Performing MOD accumulation (%=) with the right-hand side of a larger or equal type is not allowed.");
        }
    } else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_MOD == rsi::rsi_arith_accum_behaviour::NONE){
        static_assert(false, "Performing MOD accumulation (%=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    } else {
        static_assert(false, "Unrecognized behaviour.");
    }
}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_DEVICE constexpr auto& operator%=(rsi::SafeInt<T>& lhs, const U& rhs) {
    lhs %= rsi::SafeInt<U>(rhs);
    return lhs;
}

