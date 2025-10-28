
// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════


// Helper to create a minimal test tokenizer
fn create_test_tokenizer() -> NanoChatTokenizer {
    // Use GPT2 tokenizer as base
    let tokenizer_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {
                "hello": 0,
                "world": 1,
                "Ġtest": 2,
                "Ġthe": 3,
                "Ġquick": 4
            },
            "merges": []
        }
    }"#;

    NanoChatTokenizer::from_bytes(tokenizer_json.as_bytes())
        .expect("Failed to create test tokenizer")
}

#[test]
fn test_conversation_builder() {
    let messages = ConversationBuilder::new()
        .system("You are a helpful assistant")
        .user("Hello")
        .assistant("Hi there!")
        .user("How are you?")
        .build();

    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[2].role, "assistant");
    assert_eq!(messages[3].role, "user");
}

#[test]
fn test_chat_message_constructors() {
    let user = ChatMessage::user("test");
    assert_eq!(user.role, "user");
    assert_eq!(user.content, "test");

    let assistant = ChatMessage::assistant("response");
    assert_eq!(assistant.role, "assistant");

    let system = ChatMessage::system("instructions");
    assert_eq!(system.role, "system");
}

#[test]
fn test_special_tokens_default() {
    let tokens = SpecialTokens::default();
    assert_eq!(tokens.bos_token, "<|bos|>");
    assert_eq!(tokens.eos_token, "<|eos|>");
    assert_eq!(tokens.user_start, "<|user_start|>");
    assert_eq!(tokens.python_start, "<|python_start|>");
}

#[test]
fn test_tokenizer_basic_encode_decode() {
    let tokenizer = create_test_tokenizer();

    // Test basic encode
    let text = "hello world";
    let ids = tokenizer.encode(text).expect("Encoding failed");
    assert!(!ids.is_empty(), "Encoded ids should not be empty");

    // Test decode
    let decoded = tokenizer.decode(&ids).expect("Decoding failed");
    assert!(
        decoded.contains("hello") || decoded.contains("world"),
        "Decoded text should contain original words"
    );
}

#[test]
fn test_encode_with_bos() {
    let tokenizer = create_test_tokenizer();
    let text = "hello";
    let ids = tokenizer.encode_with_bos(text).expect("Encoding failed");

    assert!(!ids.is_empty());
    assert_eq!(ids[0], tokenizer.get_bos_token_id(), "First token should be BOS");
}

#[test]
fn test_special_token_registration() {
    let tokenizer = create_test_tokenizer();

    // All special tokens should be registered
    assert!(tokenizer.encode_special("<|bos|>").is_some());
    assert!(tokenizer.encode_special("<|eos|>").is_some());
    assert!(tokenizer.encode_special("<|user_start|>").is_some());
    assert!(tokenizer.encode_special("<|assistant_start|>").is_some());
    assert!(tokenizer.encode_special("<|python_start|>").is_some());

    // Non-existent token should return None
    assert!(tokenizer.encode_special("<|nonexistent|>").is_none());
}

#[test]
fn test_is_special_token() {
    let tokenizer = create_test_tokenizer();
    let bos_id = tokenizer.get_bos_token_id();

    assert!(tokenizer.is_special_token(bos_id));

    // Regular token ID should not be special
    assert!(!tokenizer.is_special_token(999));
}

#[test]
fn test_chat_template_single_message() {
    let tokenizer = create_test_tokenizer();
    let messages = vec![ChatMessage::user("Hello")];

    let tokens = tokenizer
        .apply_chat_template(&messages)
        .expect("Template application failed");

    // Should have: BOS + user_start + content + user_end
    assert!(tokens.len() >= 3, "Should have BOS, start, content, end tokens");
    assert_eq!(tokens[0], tokenizer.get_bos_token_id());
}

#[test]
fn test_chat_template_for_generation() {
    let tokenizer = create_test_tokenizer();
    let messages = vec![ChatMessage::user("Test")];

    let tokens = tokenizer
        .apply_chat_template_for_generation(&messages)
        .expect("Template failed");

    // Should end with assistant_start token
    let assistant_start = tokenizer.encode_special("<|assistant_start|>").unwrap();
    assert_eq!(
        *tokens.last().unwrap(),
        assistant_start,
        "Should end with assistant_start"
    );
}

#[test]
fn test_format_user_message() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer
        .format_user_message("Hello")
        .expect("Format failed");

    assert!(!tokens.is_empty());
    // Should include BOS, user markers, and assistant_start
    assert!(tokens.len() >= 4);
}

#[test]
fn test_format_conversation_with_system() {
    let tokenizer = create_test_tokenizer();
    let messages = vec![ChatMessage::user("Test")];

    let tokens = tokenizer
        .format_conversation(Some("You are helpful"), &messages)
        .expect("Format failed");

    // Should have system message + user message
    assert!(tokens.len() > 4);
}

#[test]
fn test_format_conversation_without_system() {
    let tokenizer = create_test_tokenizer();
    let messages = vec![ChatMessage::user("Test")];

    let tokens = tokenizer
        .format_conversation(None, &messages)
        .expect("Format failed");

    assert!(!tokens.is_empty());
}

#[test]
fn test_format_python_call() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer
        .format_python_call("2 + 2")
        .expect("Format failed");

    // Should have python_start + content + python_end
    let python_start = tokenizer.encode_special("<|python_start|>").unwrap();
    let python_end = tokenizer.encode_special("<|python_end|>").unwrap();

    assert_eq!(tokens[0], python_start);
    assert_eq!(*tokens.last().unwrap(), python_end);
}

#[test]
fn test_format_python_output() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.format_python_output("4").expect("Format failed");

    let output_start = tokenizer.encode_special("<|output_start|>").unwrap();
    let output_end = tokenizer.encode_special("<|output_end|>").unwrap();

    assert_eq!(tokens[0], output_start);
    assert_eq!(*tokens.last().unwrap(), output_end);
}

#[test]
fn test_encode_decode_batch() {
    let tokenizer = create_test_tokenizer();
    let texts = vec!["hello", "world"];

    let encoded = tokenizer.encode_batch(&texts).expect("Batch encode failed");
    assert_eq!(encoded.len(), 2);

    let ids_refs: Vec<&[u32]> = encoded.iter().map(|v| v.as_slice()).collect();
    let decoded = tokenizer.decode_batch(&ids_refs).expect("Batch decode failed");
    assert_eq!(decoded.len(), 2);
}

#[test]
fn test_vocab_size() {
    let tokenizer = create_test_tokenizer();
    let vocab_size = tokenizer.vocab_size();
    assert!(vocab_size > 0, "Vocab size should be positive");
}

#[test]
fn test_unknown_role_in_chat_template() {
    let tokenizer = create_test_tokenizer();
    let messages = vec![ChatMessage::new("unknown_role", "content")];

    let result = tokenizer.apply_chat_template(&messages);
    assert!(result.is_err(), "Should fail with unknown role");
}

#[test]
fn test_conversation_builder_push() {
    let mut builder = ConversationBuilder::new();
    builder.push(ChatMessage::user("test"));
    builder.push(ChatMessage::assistant("response"));

    let messages = builder.build();
    assert_eq!(messages.len(), 2);
}

#[test]
fn test_decode_skip_special() {
    let tokenizer = create_test_tokenizer();

    let mut tokens = vec![tokenizer.get_bos_token_id()];
    tokens.extend(tokenizer.encode("hello").unwrap());

    // Regular decode should include special tokens
    let decoded = tokenizer.decode(&tokens).unwrap();

    // Skip special should not include BOS in output
    let decoded_skip = tokenizer.decode_skip_special(&tokens).unwrap();

    // The skip version should be shorter or different
    assert!(
        decoded_skip.len() <= decoded.len(),
        "Skip special should not add extra content"
    );
}

#[test]
fn test_token_id_roundtrip() {
    let tokenizer = create_test_tokenizer();

    // Test token_to_id and id_to_token roundtrip
    if let Some(id) = tokenizer.token_to_id("hello") {
        let token = tokenizer.id_to_token(id);
        assert!(token.is_some(), "Should be able to reverse lookup");
    }
}

#[test]
fn test_special_token_map_completeness() {
    let tokenizer = create_test_tokenizer();
    let map = tokenizer.get_special_token_map();

    // Should have all 12 special tokens
    assert!(map.len() >= 12, "Should have all special tokens registered");

    // Check specific tokens
    assert!(map.contains_key("<|bos|>"));
    assert!(map.contains_key("<|python_start|>"));
    assert!(map.contains_key("<|output_end|>"));
}

