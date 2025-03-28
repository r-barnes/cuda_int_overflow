#!/usr/bin/env python3

import itertools

signs = ["", "u"]
sizes = [8] #16,32,64]

arithmetic_operators = ["+", "-", "/", "*", "%"]
accumulation_operators = ["+=", "-=", "/=", "*=", "%="]
comparisons = ["<", ">", "<=", ">=", "==", "!="]
disallow_same_size = ["*"]

class_template = """#define DEVICE

template<class T> struct LargerType {{}};
template<> struct LargerType<int8_t> {{ using type = int16_t; }};
template<> struct LargerType<uint8_t> {{ using type = int16_t; }};
template<> struct LargerType<int16_t> {{ using type = int32_t; }};
template<> struct LargerType<uint16_t> {{ using type = int32_t; }};
template<> struct LargerType<int32_t> {{ using type = int64_t; }};
template<> struct LargerType<uint32_t> {{ using type = int64_t; }};
template<> struct LargerType<int64_t> {{ using type = int64_t; }};
template<> struct LargerType<uint64_t> {{ using type = int64_t; }};

template <typename T>
struct MySafeInt {{
    static_assert(std::is_integral_v<T>, "MySafeInt can only hold integers");
    T m_value = 0;

    constexpr MySafeInt() = default;
    DEVICE constexpr explicit MySafeInt(T v) : m_value(v) {{}}

    explicit DEVICE constexpr operator T() const {{ return m_value; }}

    T value() const {{ return m_value; }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>(const MySafeInt<U> &o){{
        m_value = o.m_value;
    }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>& operator=(const MySafeInt<U> &o){{
        m_value = o.m_value;
        return *this;
    }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    MySafeInt<T>& operator=(const U &o){{
        m_value = o;
        return *this;
    }}  
}};

{non_member_operations}
"""

def get_sig(func: str) -> str:
    return func[0:func.find("{")].strip() + ";"

def make_class(type_name: str) -> str:
    return f"MySafeInt<{type_name}>"

def promote_size(s: int) -> int:
    i = sizes.index(s)
    if i < len(sizes):
        return sizes[i+1]
    else:
        print(f"Could not promote from {s}")
        return s

safe_arith = """
// Perform the arithmetic operation returning the larger type as the new type
template<typename T, typename U>
DEVICE constexpr auto operator{op}(const MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {{
    if constexpr ({not_same}){{
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value {op} rhs.m_value);
    }} else {{
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs.m_value));
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator{op}(const MySafeInt<T>& lhs, const U& rhs) {{
    if constexpr ({not_same}){{
        return MySafeInt<std::conditional_t<sizeof(T) >= sizeof(U), T, U>>(lhs.m_value {op} rhs);
    }} else {{
        static_assert(false, "Don't multiply integers of the same size together!");
        return MySafeInt<typename LargerType<T>::type>(static_cast<typename LargerType<T>::type>(lhs.m_value) + static_cast<typename LargerType<T>::type>(rhs));
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator{op}(const U& lhs, const MySafeInt<T>& rhs) {{
    return rhs {op} lhs;
}}

// Accumulation operators
template<typename T, typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U) && {not_same}>>
DEVICE constexpr auto& operator{op}=(MySafeInt<T>& lhs, const MySafeInt<U>& rhs) {{
    lhs.m_value += rhs.m_value;
    return lhs;
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U> && sizeof(T) >= sizeof(U) && {not_same}>>
DEVICE constexpr auto operator{op}=(MySafeInt<T>& lhs, const U& rhs) {{
    lhs.m_value += rhs;
    return lhs;
}}
"""

non_member_operations = []
for op in arithmetic_operators:
    not_same = "sizeof(T) != sizeof(U)" if op in disallow_same_size else "true"
    non_member_operations.append(safe_arith.format(op=op, not_same=not_same))


with open("MySafeInt.hpp", "w") as fout:
    fout.write(class_template.format(non_member_operations="\n".join(non_member_operations)))