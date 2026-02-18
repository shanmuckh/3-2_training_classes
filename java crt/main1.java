
import java.util.Scanner;

public class Main {
   public Main() {
   }

   public static void main(String[] var0) {
      System.out.println("Hello, World!");
      Scanner var1 = new Scanner(System.in);
      System.out.print("Enter your name: ");
      String var2 = var1.nextLine();
      System.out.println("Hello, " + var2 + "!");
      int var3 = var1.nextInt();
      System.out.println("The value of b is: " + var3);
      float var4 = var1.nextFloat();
      System.out.println("The value of c is: " + var4);
      double var5 = var1.nextDouble();
      System.out.println("The value of d is: " + var5);
      var1.close();
   }
}