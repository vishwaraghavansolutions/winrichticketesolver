import streamlit as st

st.set_page_config(
    page_title="System README",
    page_icon="📘",
    layout="wide"
)

st.title("📘 System Overview & Admin Guide")

st.markdown("""
Welcome to the **Ticket Management & Agent Assignment System**.  
This guide walks you through the complete workflow — from initial admin login to agents resolving tickets.

---

## 🧑‍💼 1. Admin Login

The admin begins by logging into the system using their admin credentials.  
Once authenticated, the admin is redirected to the **Admin Home** page.

---

## 🏠 2. Admin Home Page

The Admin Home page acts as the central control panel.  
From here, the admin can:

- **Manage Agents**  
- **Import Queue Data from GCP**  
- **Assign Product → Agent Resolver**  
- **Refresh Queue Data**  
- **Access Analytics Dashboard**  
- **Logout**

Each section is designed to help the admin configure the system before agents begin working.

---

## 👥 3. Setup Agents

Before any ticket assignment can happen, the admin must configure the agent list.

### What the admin can do:
- Add new agents  
- Update agent details  
- Remove inactive agents  
- Assign roles (if applicable)  

This ensures that the system knows which agents exist and can route tickets correctly.

---

## ☁️ 4. Import Queue Data from GCP

Once agents are configured, the admin imports the latest ticket queue.

### The import process:
- Connect to GCP storage  
- Fetch the Parquet or CSV ticket dataset  
- Load it into the system  
- Convert timestamps and normalize fields  
- Store the processed dataset locally or in S3  

After import, the admin can refresh the queue anytime to pull updated data.

---

## 🧩 5. Assign Resolver (Product → Agent Mapping)

This is a critical step.

Each product category must be mapped to an agent so the system knows **who handles which tickets**.

### Example:
| Product Name        | Assigned Agent |
|---------------------|----------------|
| Mutual Funds        | agent_2        |
| Insurance           | agent_3        |
| SIP                 | agent_1        |

The admin uses the Resolver Editor to:
- View all products  
- Assign or reassign agents  
- Save resolver mappings  

Once saved, the system automatically attaches `agent_id` to every ticket.

---

## 🔐 6. Agents Login & Password Setup

After the admin completes setup, agents are invited to log in.

### First-time login flow:
1. Agent enters their email/ID  
2. System detects first login  
3. Agent is prompted to **set a new password**  
4. Password is securely stored  
5. Agent is redirected to their **Agent Dashboard**

This ensures secure onboarding for every agent.

---

## 🎟️ 7. Agents Start Working on Tickets

Once logged in, agents can:

- View tickets assigned to them  
- Filter by status, sentiment, SLA risk  
- Respond to customers  
- Update ticket status  
- Track their performance metrics  

The system ensures each agent only sees **their own tickets**, based on the resolver mapping.

---

## 📊 8. Analytics & Monitoring

Admins can access the analytics dashboard to monitor:

- Ticket volume  
- SLA breaches  
- Sentiment trends  
- Agent performance  
- Product-level insights  
- Drill-down ticket details  

This helps admins optimize staffing, training, and product workflows.

---

## 🚀 Summary

The full workflow looks like this:

1. **Admin logs in**  
2. **Admin configures agents**  
3. **Admin imports ticket data from GCP**  
4. **Admin assigns resolver (product → agent)**  
5. **Agents log in and set passwords**  
6. **Agents work on assigned tickets**  
7. **Admin monitors analytics**

This README page ensures every admin understands the system end-to-end.

---

If you need a more visual version (flowchart, icons, collapsible sections), I can enhance this page further.
""")