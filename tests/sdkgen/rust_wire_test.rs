use headroom_generated_pilot::{Client, Error, Options, RetrieveRequest};
use serde_json::{Map, Value};

fn client() -> Client {
    Client::new(
        &std::env::var("HEADROOM_WIRE_TEST_URL").expect("HEADROOM_WIRE_TEST_URL"),
        Options::default(),
    )
    .expect("valid fixture URL")
}

#[tokio::test]
async fn post_and_get_preserve_wire_values() {
    let response = client()
        .retrieve(&RetrieveRequest {
            hash: "ok".into(),
            additional_properties: Map::from_iter([(
                "extra_request".into(),
                Value::String("unchanged".into()),
            )]),
        })
        .await
        .expect("POST retrieval succeeds");
    assert_eq!(response.tool_name, None);
    assert_eq!(response.original_content, r#"{"snake_case":"世界"}"#);
    assert_eq!(response.additional_properties["future_extension"]["snake_case"], "unchanged");

    let key = "a/世界 ?#+'!*()";
    let by_hash = client().retrieve_get(key).await.expect("GET retrieval succeeds");
    assert_eq!(by_hash.hash, key);
}

#[tokio::test]
async fn http_error_retains_body_without_displaying_it() {
    let error = client()
        .retrieve(&RetrieveRequest {
            hash: "missing".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect_err("missing entry returns an API error");
    match error {
        Error::API(api) => {
            assert_eq!(api.status, 404);
            assert!(String::from_utf8_lossy(&api.body).contains("Entry missing"));
            assert!(!api.to_string().contains("Entry missing"));
        }
        other => panic!("expected API error, got {other}"),
    }
}
