using System.Text.Json;
using Headroom.GeneratedPilot;

const string valid = """
    {"hash":"h","original_content":"x","original_tokens":1,"original_item_count":1,"compressed_item_count":1,"tool_name":null,"retrieval_count":1,"future_extension":{"snake_case":"unchanged"}}
    """;
var parsed = JsonSerializer.Deserialize<RetrieveResponse>(valid)
    ?? throw new Exception("Explicit-null response decoded to null");
if (parsed.ToolName is not null)
    throw new Exception("Explicit null was not preserved");
if (parsed.AdditionalProperties["future_extension"].GetProperty("snake_case").GetString() != "unchanged")
    throw new Exception("Unknown response property was not preserved");

const string missing = """
    {"hash":"h","original_content":"x","original_tokens":1,"original_item_count":1,"compressed_item_count":1,"retrieval_count":1}
    """;
try
{
    JsonSerializer.Deserialize<RetrieveResponse>(missing);
    throw new Exception("Missing required-nullable tool_name was accepted");
}
catch (JsonException)
{
    // Expected: required nullable means the property must be present, though its value may be null.
}

Console.WriteLine("PASS: .NET required-nullable and unknown-field model checks");
