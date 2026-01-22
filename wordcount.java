public class wordcount {
    public static void main(String[] args) {
        String text = "Java is easy and powerful";

        int wordCount = text.trim().split("\\s+").length;

        System.out.println("Word count: " + wordCount);
    }
}
