import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("SGD with Momentum and Contour Plot")

    # Input for initial parameter, learning rate, and momentum
    initial_x = st.text_input("Initial X:", "2.0")
    initial_y = st.text_input("Initial Y:", "2.0")
    learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, step=0.01)
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, step=0.01)

    # Initialize variables using st.session_state
    st.session_state.x_param = st.session_state.get("x_param", float(initial_x))
    st.session_state.y_param = st.session_state.get("y_param", float(initial_y))
    st.session_state.x_velocity = st.session_state.get("x_velocity", 0)
    st.session_state.y_velocity = st.session_state.get("y_velocity", 0)
    st.session_state.step = st.session_state.get("step", 0)

    if "param_history" not in st.session_state:
        st.session_state.param_history = []

    if st.button("Next Step"):
        # Compute the gradients (you can replace this with your own function)
        grad_x = 2 * st.session_state.x_param
        grad_y = 2 * st.session_state.y_param
        st.session_state.x_velocity = momentum * st.session_state.x_velocity - learning_rate * grad_x
        st.session_state.y_velocity = momentum * st.session_state.y_velocity - learning_rate * grad_y
        st.session_state.x_param += st.session_state.x_velocity
        st.session_state.y_param += st.session_state.y_velocity
        st.session_state.step += 1
        # Append the current parameter values to the history
        st.session_state.param_history.append((st.session_state.x_param, st.session_state.y_param))

    # Create a grid for the contour plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Display the parameter updates
    st.header("SGD with Momentum Parameter Updates:")
    # st.write(f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}")
    # st.write(f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}")
    st.text(f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}")


    # Plot contour and parameter updates
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 2, 20), cmap='viridis')
    
    # Plot the parameter path as a line
    if len(st.session_state.param_history) > 1:
        param_history = np.array(st.session_state.param_history)
        plt.plot(param_history[:, 0], param_history[:, 1], marker='o', linestyle='-', color='r', label='Parameter Path')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()

