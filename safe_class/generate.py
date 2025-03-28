#!/usr/bin/env python3

import itertools
from dataclasses import dataclass

signs = ["", "u"]
sizes = [8] #16,32,64]

@dataclass(frozen=True)
class Op:
    name: str
    op: str

arithmetic_operators = [
    Op("ADD", "+" ),
    Op("SUB", "-" ),
    Op("DIV", "/" ),
    Op("MUL", "*" ),
    Op("MOD", "%")
]


accumulation_operators = ["+=", "-=", "/=", "*=", "%="]
comparisons = ["<", ">", "<=", ">=", "==", "!="]
disallow_same_size = ["*"]

class_template = """// Richard's Safe Int Class
#define DEVICE

namespace rsi {{

namespace detail {{

enum class rsi_arith_behaviour {{
    STANDARD_BEHAVIOUR,
    STANDARD_BUT_NOT_SAME_SIZE,
    SAME_SIZE_PROMOTE_TO_LARGER_SIGNED,
    NONE
}};

enum class rsi_accum_behaviour {{
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL,
    ONLY_SMALLER,
    NONE
}};

{behaviour_defaults}

template<class T> struct LargerType {{}};
template<> struct LargerType<bool> {{ using type = int32_t; }};
template<> struct LargerType<int8_t> {{ using type = int32_t; }};
template<> struct LargerType<uint8_t> {{ using type = int32_t; }};
template<> struct LargerType<int16_t> {{ using type = int32_t; }};
template<> struct LargerType<uint16_t> {{ using type = int32_t; }};
template<> struct LargerType<int32_t> {{ using type = int64_t; }};
template<> struct LargerType<uint32_t> {{ using type = int64_t; }};
template<> struct LargerType<int64_t> {{ using type = int64_t; }};
template<> struct LargerType<uint64_t> {{ using type = int64_t; }};

}}

template <typename T>
class SafeInt {{
  private:
    static_assert(std::is_integral_v<T>, "SafeInt can only hold integers");
    T m_value = 0;

  public:

    constexpr SafeInt() = default;
    DEVICE constexpr explicit SafeInt(T v) : m_value(v) {{}}

    explicit DEVICE constexpr operator T() const {{ return m_value; }}

    T value() const {{ return m_value; }}
    T& value() {{ return m_value; }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>(const SafeInt<U> &o){{
        m_value = o.m_value;
    }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>& operator=(const SafeInt<U> &o){{
        m_value = o.m_value;
        return *this;
    }}

    template<typename U, typename = std::enable_if_t<sizeof(T) >= sizeof(U)>>
    SafeInt<T>& operator=(const U &o){{
        m_value = o;
        return *this;
    }}  
}};

}}

{non_member_operations}
"""

def get_sig(func: str) -> str:
    return func[0:func.find("{")].strip() + ";"

def make_class(type_name: str) -> str:
    return f"SafeInt<{type_name}>"

def promote_size(s: int) -> int:
    i = sizes.index(s)
    if i < len(sizes):
        return sizes[i+1]
    else:
        print(f"Could not promote from {s}")
        return s

safe_arith = """
// {op_name} arithmetic operations
template<typename T, typename U>
DEVICE constexpr auto operator{op}(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {{
    if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_{op_name} == rsi::detail::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {{
        return rsi::SafeInt(lhs.value() * rhs.value());
    }} else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_{op_name} == rsi::detail::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE) {{
        if constexpr (sizeof(T) == sizeof(U)) {{
            static_assert(false, "Performing {op_name} ({op}) on types of the same size is not allowed.");
        }} else {{
            return rsi::SafeInt(lhs.value() * rhs.value());
        }}
    }} else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_{op_name} == rsi::detail::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED){{
        if constexpr (sizeof(T) == sizeof(U)) {{
            using larger = rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) {op} static_cast<larger>(rhs.value()));
        }} else {{
            return rsi::SafeInt(lhs.value() * rhs.value());
        }}
    }} else if constexpr (rsi::detail::RSI_ARITH_BEHAVIOUR_{op_name} == rsi::detail::rsi_arith_behaviour::NONE){{
        static_assert(false, "Performing {op_name} is not allowed.");
    }} else {{
        static_assert(false, "Unrecognized behaviour.");
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator{op}(const rsi::SafeInt<T>& lhs, const U& rhs) {{
    return lhs {op} rsi::SafeInt(rhs);
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto operator{op}(const U& lhs, const rsi::SafeInt<T>& rhs) {{
    return rhs {op} rsi::SafeInt(lhs);
}}

// {op_name} accumulation operators
template<typename T, typename U>
DEVICE constexpr auto& operator{op}=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {{
    if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_{op_name} == rsi::detail::rsi_accum_behaviour::STANDARD_BEHAVIOUR) {{
        return (lhs.value() += rhs.value());
    }} else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_{op_name} == rsi::detail::rsi_accum_behaviour::SMALLER_OR_EQUAL) {{
        if constexpr (sizeof(T) >= sizeof(U)) {{
            return lhs.value() += rhs.value();
        }} else {{
            static_assert(false, "Performing {op_name} accumulation ({op}=) with the right-hand side of a larger type is not allowed.");
        }}
    }} else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_{op_name} == rsi::detail::rsi_accum_behaviour::ONLY_SMALLER){{
        if constexpr (sizeof(T) > sizeof(U)) {{
            return lhs.value() += rhs.value();
        }} else {{
            static_assert(false, "Performing {op_name} accumulation ({op}=) with the right-hand side of a larger or equal type is not allowed.");
        }}
    }} else if constexpr (rsi::detail::RSI_ACCUM_BEHAVIOUR_{op_name} == rsi::detail::rsi_accum_behaviour::NONE){{
        static_assert(false, "Performing {op_name} is not allowed.");
    }} else {{
        static_assert(false, "Unrecognized behaviour.");
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
DEVICE constexpr auto& operator{op}=(rsi::SafeInt<T>& lhs, const U& rhs) {{
    lhs += SafeInt(rhs);
    return lhs;
}}
"""

arith_behaviour_default_template = "constexpr rsi_arith_behaviour RSI_ARITH_BEHAVIOUR_{op_name} = rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED;"
accum_behaviour_default_template = "constexpr rsi_accum_behaviour RSI_ACCUM_BEHAVIOUR_{op_name} = rsi_accum_behaviour::STANDARD_BEHAVIOUR;"

non_member_operations = []
behaviour_defaults = []
for op in arithmetic_operators:
    non_member_operations.append(safe_arith.format(op_name=op.name, op=op.op))
    behaviour_defaults.append(arith_behaviour_default_template.format(op_name=op.name))
    behaviour_defaults.append(accum_behaviour_default_template.format(op_name=op.name))

with open("MySafeInt.hpp", "w") as fout:
    fout.write(class_template.format(
        behaviour_defaults="\n".join(behaviour_defaults), 
        non_member_operations="\n".join(non_member_operations)))