# temp.py
import os


from src.category_matcher.category_matcher import CategoryMatcher

def test_category_matching():
    """Test the category matcher with various queries"""
    
    # Initialize the category matcher
    # Make sure your ChatBot Queries.xlsx is in the data directory
    matcher = CategoryMatcher('data\ChatBot.xlsx')
    
    # Test cases - various IT support queries
    test_queries = [
        "Need more resources for VM",
        "My laptop is running very slow",
        "Cannot connect to VPN",
        "Printer is not printing anything",
        "Need access to SharePoint files",
        "Server is not responding",
        "Wi-Fi keeps disconnecting",
        "Need to provision new virtual machines",
        "Backup restore is failing",
        "Email is not working",
        "Teams video call issues",
        "Need Azure resource group access",
        "Firewall configuration problems",
        "SSL certificate expired",
        "Network switch port not working",
        "Need new keyboard and mouse",
        "Software installation failing",
        "Adobe license renewal needed",
        "Database query performance issues",
        "Power BI report generation failed"
    ]
    
    print("IT Support Category Matcher - Test Cases")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Case {i}: '{query}'")
        print("-" * 40)
        
        # Find top 5 categories
        top_categories = matcher.find_top_categories(query, top_k=5)
        
        if top_categories:
            print("Top 5 matching categories:")
            for j, cat in enumerate(top_categories, 1):
                print(f"  {j}. {cat['category']} (similarity: {cat['similarity']:.4f})")
        else:
            print("No matching categories found.")
            # Show available categories for debugging
            if hasattr(matcher, 'categories') and matcher.categories:
                print("Available categories (first 5):")
                for k, cat in enumerate(matcher.categories[:5], 1):
                    print(f"  {k}. {cat}")
    
    # Test specific query that was mentioned in the issue
    print(f"\n\nSpecific Test Case: 'Need more resources for VM'")
    print("-" * 50)
    specific_query = "Need more resources for VM"
    top_categories = matcher.find_top_categories(specific_query, top_k=5)
    
    if top_categories:
        print("Top 5 matching categories:")
        for j, cat in enumerate(top_categories, 1):
            print(f"  {j}. {cat['category']} (similarity: {cat['similarity']:.4f})")
            print(f"      Q1: {cat['q1']}")
            print(f"      Q2: {cat['q2']}")
            print(f"      Q2a: {cat['q2a']}")
    else:
        print("No matching categories found for 'Need more resources for VM'")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n\nEdge Case Testing")
    print("=" * 30)
    
    # Test with empty query
    matcher = CategoryMatcher('data\ChatBot.xlsx')
    
    edge_cases = [
        "",  # Empty query
        "xyz123",  # Random characters
        "Need help with something",  # Very generic query
        "This is a very long query with many words that might not match any specific category in the knowledge base"  # Long query
    ]
    
    for i, query in enumerate(edge_cases, 1):
        print(f"\nEdge Case {i}: '{query}'")
        top_categories = matcher.find_top_categories(query, top_k=3)
        if top_categories:
            print(f"  Found {len(top_categories)} categories")
            for j, cat in enumerate(top_categories, 1):
                print(f"    {j}. {cat['category']} ({cat['similarity']:.4f})")
        else:
            print("  No categories found")

def test_model_persistence():
    """Test that the model saves and loads correctly"""
    print("\n\nModel Persistence Test")
    print("=" * 25)
    
    model_path = "models/test_category_matcher.pkl"
    
    # Create and save model
    matcher1 = CategoryMatcher('data\ChatBot.xlsx', model_path)
    print(f"Model saved to {model_path}")
    
    # Load model
    matcher2 = CategoryMatcher('data\ChatBot.xlsx', model_path)
    print("Model loaded successfully")
    
    # Test both models with same query
    test_query = "VM provisioning issues"
    results1 = matcher1.find_top_categories(test_query, top_k=3)
    results2 = matcher2.find_top_categories(test_query, top_k=3)
    
    print(f"Results from first model: {len(results1)} categories")
    print(f"Results from second model: {len(results2)} categories")
    
    # Clean up test file
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Test model file cleaned up")

if __name__ == "__main__":
    print("Running Category Matcher Tests...")
    
    # Run all tests
    test_category_matching()
    test_edge_cases()
    test_model_persistence()
    
    print("\n\nAll tests completed!")