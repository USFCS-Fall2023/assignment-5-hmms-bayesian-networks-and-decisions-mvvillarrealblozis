from pgmpy.inference import VariableElimination
import carnet

print("\nQuestion 2\n")

from HMM import HMM, Observation
model = HMM()
model.load('partofspeech.browntags.trained')

generate = model.generate(20)  # n=10

print("Generated State Sequence:", ' '.join(generate.stateseq))
print("Generated Emission Sequence:", ' '.join(generate.outputseq))

obs = Observation([], generate.outputseq)

forward_prob = model.forward(' '.join(obs.outputseq))
print("Forward probability of the observation sequence:", forward_prob)

viterbi_path = model.viterbi(obs)
print("Most likely sequence of states (Viterbi):", ' '.join(viterbi_path))

print("\nQuestion 3")

print("\nALARM.PY")

from alarm import alarm_model
alarm_infer = VariableElimination(alarm_model)

print("\nProbability of Mary Calling given that John called")
print(alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"}))

print("\nProbability of both John and Mary calling given Alarm")
print(alarm_infer.query(variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"}))

print("\nProbability of Alarm given that Mary called")
print(alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls": "yes"}))

print("\nCARNET.PY")

carnet.car_model.add_cpds(carnet.cpd_battery, carnet.cpd_gas, carnet.cpd_radio, carnet.cpd_ignition,
                          carnet.cpd_key_present, carnet.cpd_starts, carnet.cpd_moves)

car_infer = VariableElimination(carnet.car_model)

print("\nGiven that the car will not move, what is the probability that the battery is not working?")
print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))

print("\nGiven that the radio is not working, what is the probability that the car will not start?")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))

print("\nQuery 3.1: Given that the battery is working, what is the probability of the radio working?")
print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works"}))

print("\nQuery 3.2: Given that the battery is working and the car has gas, what is the probability of the radio working?")
print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))

print("\nQuery 4.1: Given that the car doesSUB 't move, what is the probability of the ignition failing?")
print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no"}))

print("\nQuery 4.2: Given that the car doesn't move and has no gas, what is the probability of the ignition failing?")
print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))

print("\nQuery 5: What is the probability that the car starts if the radio works and it has gas in it?")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))