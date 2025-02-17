


class Cost(object):
    def __init__(self):
        self.accumulated_energy = 0
        self.accumulated_latency = 0


    def get_latency_energy(self, selected_clients, round_i):
        latency_sum = self.get_latency_sum(selected_clients, round_i)
        energy_sum = self.get_energy_sum(selected_clients, round_i)
        return (latency_sum, energy_sum)
    
    def get_latency_sum(self, selected_clients, round_i):
        latency_sum = 0

        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalDelay(round_i)
            upload[i] = selected_clients[i].getUploadDelay(round_i)
            if local[i] + upload[i] > latency_sum:
                # print("local[i] + upload[i] ", local[i],upload[i] )
                latency_sum = local[i] + upload[i]

        return latency_sum

        
    def get_energy_sum(self, selected_clients, round_i):
        energy_sum = 0
        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalEnergy(round_i)
            upload[i] = selected_clients[i].getUploadEnergy(round_i)
            energy_sum += (local[i] + upload[i])
        return energy_sum
            



class Clients_System_Attr(object):
    def __init__(self, cpu_frequency, transmit_rate, transmit_power):
        self.cpu_frequency = cpu_frequency
        self.transmit_rate,= transmit_rate,
        self.transmit_power = transmit_power

    def get_client_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "transmit_rate": self.transmit_rate[id],
            "transmit_power": self.transmit_power[id],
        }
