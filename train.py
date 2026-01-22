from daytona import Daytona
from dotenv import load_dotenv

load_dotenv()

daytona = Daytona()
sandbox = daytona.create()

try:
    response = sandbox.process.code_run(
        'print("Sum of 3 and 4 is " + str(3 + 4))'
    )

    if response.exit_code != 0:
        print("Error:", response.result)
    else:
        print(response.result)

    exec_response = sandbox.process.exec("echo 'Hello, World!'")
    print(exec_response.result)

finally:
    daytona.delete(sandbox)
