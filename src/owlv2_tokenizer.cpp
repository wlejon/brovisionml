// OWLv2's CLIP tokenizer: byte-level BPE over HF vocab.json + merges.txt.
//
// The same algorithm as HF CLIPTokenizer on ASCII text: lowercase, collapse
// whitespace, split into pre-tokens with CLIP's pattern
//   's | 't | 're | 've | 'm | 'll | 'd | letters+ | one digit | other non-space+
// map each pre-token's bytes into GPT-2's printable-unicode alphabet, mark the
// last symbol with "</w>", and merge adjacent pairs lowest rank first.
// Non-ASCII bytes are carried through as part of a punctuation run (still
// byte-encoded, so nothing is lost; HF would split them by Unicode class).

#include "brovisionml/owlv2.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brovisionml::owlv2 {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("owlv2::Tokenizer: " + msg);
}

void put_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Split UTF-8 into code points (each kept as its UTF-8 bytes).
std::vector<std::string> code_points(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t n = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

// The contraction suffix starting at text[i], or 0.
std::size_t contraction(std::string_view t, std::size_t i) {
    if (i >= t.size() || t[i] != '\'') return 0;
    for (const char* s : {"'re", "'ve", "'ll"})
        if (t.compare(i, 3, s) == 0) return 3;
    for (const char* s : {"'s", "'t", "'m", "'d"})
        if (t.compare(i, 2, s) == 0) return 2;
    return 0;
}

std::vector<std::string> pre_tokenize(std::string_view t) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < t.size()) {
        const unsigned char c = static_cast<unsigned char>(t[i]);
        if (is_space(c)) { ++i; continue; }
        if (std::size_t n = contraction(t, i)) { out.emplace_back(t.substr(i, n)); i += n; continue; }
        std::size_t j = i + 1;
        if (is_letter(c)) {
            while (j < t.size() && is_letter(static_cast<unsigned char>(t[j]))) ++j;
        } else if (!is_digit(c)) {
            while (j < t.size()) {
                const unsigned char u = static_cast<unsigned char>(t[j]);
                if (is_space(u) || is_letter(u) || is_digit(u) || contraction(t, j)) break;
                ++j;
            }
        }
        out.emplace_back(t.substr(i, j - i));
        i = j;
    }
    return out;
}

// vocab.json is a flat {"token": id, ...} object.
std::unordered_map<std::string, int32_t> read_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    const char* p = text.data();
    const char* end = p + text.size();
    auto ws = [&] { while (p < end && is_space(static_cast<unsigned char>(*p))) ++p; };
    auto expect = [&](char c) { ws(); if (p >= end || *p != c) fail(std::string("vocab.json: expected '") + c + "'"); ++p; };
    std::unordered_map<std::string, int32_t> vocab;
    expect('{');
    ws();
    if (p < end && *p == '}') return vocab;
    while (true) {
        expect('"');
        std::string key;
        while (p < end && *p != '"') {
            if (*p != '\\') { key += *p++; continue; }
            if (p + 1 >= end) fail("vocab.json: truncated escape");
            const char e = p[1];
            p += 2;
            switch (e) {
                case 'n': key += '\n'; break;
                case 't': key += '\t'; break;
                case 'r': key += '\r'; break;
                case 'b': key += '\b'; break;
                case 'f': key += '\f'; break;
                case 'u': {
                    if (end - p < 4) fail("vocab.json: truncated \\u escape");
                    uint32_t cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char h = p[k];
                        const int v = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10
                                    : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                        if (v < 0) fail("vocab.json: bad \\u escape");
                        cp = cp * 16 + static_cast<uint32_t>(v);
                    }
                    p += 4;
                    put_utf8(cp, key);
                    break;
                }
                default: key += e; break;   // \" \\ \/
            }
        }
        if (p >= end) fail("vocab.json: unterminated string");
        ++p;
        expect(':');
        ws();
        long v = 0;
        if (p >= end || !is_digit(static_cast<unsigned char>(*p))) fail("vocab.json: expected an id");
        while (p < end && is_digit(static_cast<unsigned char>(*p))) v = v * 10 + (*p++ - '0');
        vocab.emplace(std::move(key), static_cast<int32_t>(v));
        ws();
        if (p < end && *p == ',') { ++p; continue; }
        expect('}');
        break;
    }
    return vocab;
}

std::unordered_map<std::string, int32_t> read_merges(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    std::unordered_map<std::string, int32_t> ranks;
    std::string line;
    int32_t rank = 0;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first && line[0] == '#') { first = false; continue; }   // "#version: ..."
        first = false;
        const auto sp = line.find(' ');
        if (sp == std::string::npos) fail("merges.txt: malformed line '" + line + "'");
        ranks.emplace(line.substr(0, sp) + '\x01' + line.substr(sp + 1), rank++);
    }
    return ranks;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& vocab_json, const std::string& merges_txt) {
    Tokenizer t;
    t.vocab_ = read_vocab(vocab_json);
    t.ranks_ = read_merges(merges_txt);
    // GPT-2's byte -> printable code point table: printable Latin-1 bytes map
    // to themselves, the rest to 256, 257, ... in byte order.
    int next = 256;
    for (int b = 0; b < 256; ++b) {
        const bool self = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        put_utf8(self ? static_cast<uint32_t>(b) : static_cast<uint32_t>(next++), t.byte_to_unicode_[b]);
    }
    if (t.vocab_.size() < 256 || t.ranks_.empty()) fail("vocab.json / merges.txt look empty");
    return t;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, int max_len) const {
    // Lowercase (ASCII) and collapse whitespace runs.
    std::string clean;
    clean.reserve(text.size());
    bool space = true;
    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (is_space(c)) {
            if (!space) clean += ' ';
            space = true;
            continue;
        }
        space = false;
        clean += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }

    // OWLv2's release registers its pad token "!" (id 0) as an added token,
    // and HF splits added tokens out of the text before BPE: every "!" is id
    // 0, and the text between them is tokenised piece by piece.
    std::vector<std::string> pieces;
    std::size_t from = 0;
    for (std::size_t bang = clean.find('!');; bang = clean.find('!', from)) {
        const std::string chunk = clean.substr(from, bang == std::string::npos ? std::string::npos : bang - from);
        for (std::string& p : pre_tokenize(chunk)) pieces.push_back(std::move(p));
        if (bang == std::string::npos) break;
        pieces.emplace_back("!");
        from = bang + 1;
    }

    std::vector<int32_t> out{kBos};
    for (const std::string& piece : pieces) {
        if (piece == "!") { out.push_back(kBang); continue; }
        std::string mapped;
        for (char c : piece) mapped += byte_to_unicode_[static_cast<unsigned char>(c)];
        std::vector<std::string> word = code_points(mapped);
        word.back() += "</w>";
        while (word.size() > 1) {
            int best = -1;
            std::size_t at = 0;
            for (std::size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = ranks_.find(word[i] + '\x01' + word[i + 1]);
                if (it != ranks_.end() && (best < 0 || it->second < best)) { best = it->second; at = i; }
            }
            if (best < 0) break;
            const std::string a = word[at], b = word[at + 1];
            std::vector<std::string> next;
            next.reserve(word.size());
            for (std::size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                    next.push_back(a + b);
                    i += 2;
                } else {
                    next.push_back(word[i++]);
                }
            }
            word = std::move(next);
        }
        for (const std::string& u : word) {
            auto it = vocab_.find(u);
            if (it != vocab_.end()) out.push_back(it->second);
        }
    }
    if (max_len < 2) max_len = 2;
    if (static_cast<int>(out.size()) > max_len - 1) out.resize(static_cast<std::size_t>(max_len - 1));
    out.push_back(kEos);
    return out;
}

}  // namespace brovisionml::owlv2
